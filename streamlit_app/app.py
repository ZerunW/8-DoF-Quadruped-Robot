import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd
import time

# ==================== Page Config ====================
st.set_page_config(
    page_title="Quadruped Robot Leg Kinematics - Full Interactive",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Session State ====================
if 'motor1' not in st.session_state:
    st.session_state.motor1 = 90
if 'motor2' not in st.session_state:
    st.session_state.motor2 = 90
if 'trajectory_path' not in st.session_state:
    st.session_state.trajectory_path = None
if 'workspace_boundary' not in st.session_state:
    st.session_state.workspace_boundary = None
if 'gait_time' not in st.session_state:
    st.session_state.gait_time = 0
if 'gait_playing' not in st.session_state:
    st.session_state.gait_playing = False
if 'motor_table' not in st.session_state:
    st.session_state.motor_table = []
if 'target_pos' not in st.session_state:
    st.session_state.target_pos = None

# Initialize parameters with your defaults
if 'params' not in st.session_state:
    st.session_state.params = {
        'd_x': -20, 'd_y': -20,
        'L1': 110, 'L2': 110, 'L2a': 30, 'L3': 24,
        'L41': 24, 'L42': 30, 'phi': 90.05,
        'L51': 28.2, 'L52': 110
    }

if 'gait_params' not in st.session_state:
    st.session_state.gait_params = {
        'step_length': 20,
        'step_height': 10,
        'period': 1.0,
        'center_x': -10.0,  # 改为 float
        'center_y': -155.7
    }

# ==================== Kinematics Functions ====================

def calculate_kinematics(motor_angle1, motor_angle2, params):
    """Forward kinematics with custom parameters"""
    d_x, d_y = params['d_x'], params['d_y']
    L1, L2, L2a, L3 = params['L1'], params['L2'], params['L2a'], params['L3']
    L41, L42, phi = params['L41'], params['L42'], params['phi']
    L51, L52 = params['L51'], params['L52']
    
    alpha = 5*np.pi/4 + np.deg2rad(motor_angle1)
    gamma = np.pi/4 + np.deg2rad(motor_angle2)
    
    O = np.array([0, 0])
    M = np.array([d_x, d_y])
    K = np.array([L1*np.cos(alpha), L1*np.sin(alpha)])
    C = np.array([d_x + L3*np.cos(gamma), d_y + L3*np.sin(gamma)])
    
    C_len = np.linalg.norm(C)
    k1 = (C_len**2 + L41**2 - L51**2) / (2*L41)
    k1 = np.clip(k1, -C_len, C_len)
    angle_C = np.arctan2(C[1], C[0])
    d1 = angle_C + np.arccos(k1/C_len)
    d2 = angle_C - np.arccos(k1/C_len)
    D1 = np.array([L41*np.cos(d1), L41*np.sin(d1)])
    D2 = np.array([L41*np.cos(d2), L41*np.sin(d2)])
    delta = d1 if D1[1] > D2[1] else d2
    D = D1 if D1[1] > D2[1] else D2
    
    delta_A = delta - np.deg2rad(phi)
    A = np.array([L42*np.cos(delta_A), L42*np.sin(delta_A)])
    
    v = A - K
    v_len = np.linalg.norm(v)
    k2 = (v_len**2 + L2a**2 - L52**2) / (2*L2a)
    k2 = np.clip(k2, -v_len, v_len)
    angle_v = np.arctan2(v[1], v[0])
    bb1 = angle_v + np.arccos(k2/v_len)
    bb2 = angle_v - np.arccos(k2/v_len)
    
    B1 = K + L2a*np.array([np.cos(bb1), np.sin(bb1)])
    B2 = K + L2a*np.array([np.cos(bb2), np.sin(bb2)])
    
    def crosses(A, B, O, K):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        if np.allclose(A,O) or np.allclose(A,K) or np.allclose(B,O) or np.allclose(B,K):
            return False
        return ccw(A,O,K) != ccw(B,O,K) and ccw(A,B,O) != ccw(A,B,K)
    
    c1 = crosses(A, B1, O, K)
    c2 = crosses(A, B2, O, K)
    
    if c1 and not c2:
        B, beta_b = B2, bb2
    elif c2 and not c1:
        B, beta_b = B1, bb1
    else:
        d1_dist = abs(np.linalg.norm(A - B1) - L52)
        d2_dist = abs(np.linalg.norm(A - B2) - L52)
        B, beta_b = (B1, bb1) if d1_dist < d2_dist else (B2, bb2)
    
    beta_prime = beta_b + np.pi
    E = K + L2*np.array([np.cos(beta_prime), np.sin(beta_prime)])
    
    alpha_deg = np.rad2deg(alpha) % 360
    angle_KOX = alpha_deg if alpha_deg <= 180 else 360 - alpha_deg
    
    vec_thigh = K - O
    vec_calf = E - K
    cos_theta = np.dot(vec_thigh, vec_calf) / (np.linalg.norm(vec_thigh) * np.linalg.norm(vec_calf))
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_OKE = np.rad2deg(np.arccos(cos_theta))

    return {
        'points': {'O': O, 'M': M, 'K': K, 'C': C, 'D': D, 'A': A, 'B': B, 'E': E},
        'angles': {'KOX': angle_KOX, 'OKE': angle_OKE}
    }

def inverse_kinematics(target_x, target_y, initial_guess, params):
    """Inverse kinematics"""
    def objective(angles):
        m1, m2 = angles
        if m1 < 60 or m1 > 120 or m2 < 60 or m2 > 100:
            return 1e10
        try:
            result = calculate_kinematics(m1, m2, params)
            E = result['points']['E']
            error = (E[0] - target_x)**2 + (E[1] - target_y)**2
            return error
        except:
            return 1e10
    
    result = minimize(objective, initial_guess, method='Nelder-Mead',
                     options={'xatol': 0.01, 'fatol': 0.01, 'maxiter': 1000})
    
    if result.success and result.fun < 1.0:
        return result.x[0], result.x[1], True
    else:
        for guess in [(80, 80), (100, 90), (90, 70), (70, 90)]:
            result = minimize(objective, guess, method='Nelder-Mead',
                            options={'xatol': 0.01, 'fatol': 0.01, 'maxiter': 1000})
            if result.success and result.fun < 1.0:
                return result.x[0], result.x[1], True
        return result.x[0], result.x[1], False

def foot_trajectory(t, step_length, step_height, period, center_x, center_y):
    """Generate foot trajectory for walking gait"""
    phase = (t % period) / period
    
    if phase < 0.5:
        swing_phase = phase * 2
        x_offset = step_length/2 - step_length * swing_phase
        y_offset = step_height * np.sin(np.pi * swing_phase)
    else:
        stance_phase = (phase - 0.5) * 2
        x_offset = -step_length/2 + step_length * stance_phase
        y_offset = 0
    
    x = center_x + x_offset
    y = center_y + y_offset
    return x, y

def generate_trajectory_path(step_length, step_height, period, center_x, center_y, num_points=100):
    """Generate complete trajectory path"""
    t_array = np.linspace(0, period, num_points)
    path = [foot_trajectory(t, step_length, step_height, period, center_x, center_y) 
            for t in t_array]
    return np.array(path)

@st.cache_data
def calculate_workspace_boundary(params, resolution=2):
    """Calculate workspace boundary"""
    boundary_points = []
    
    # Trace boundary by varying motor angles
    for m2 in range(60, 101, resolution):
        try:
            data = calculate_kinematics(60, m2, params)
            E = data['points']['E']
            boundary_points.append([E[0], E[1]])
        except:
            pass
    
    for m1 in range(60, 121, resolution):
        try:
            data = calculate_kinematics(m1, 100, params)
            E = data['points']['E']
            boundary_points.append([E[0], E[1]])
        except:
            pass
    
    for m2 in range(100, 59, -resolution):
        try:
            data = calculate_kinematics(120, m2, params)
            E = data['points']['E']
            boundary_points.append([E[0], E[1]])
        except:
            pass
    
    for m1 in range(120, 59, -resolution):
        try:
            data = calculate_kinematics(m1, 60, params)
            E = data['points']['E']
            boundary_points.append([E[0], E[1]])
        except:
            pass
    
    return np.array(boundary_points)

# ==================== Plotly Plotting ====================

def create_plotly_figure(data, workspace_boundary=None, trajectory_path=None, target_pos=None):
    """Create interactive Plotly figure"""
    points = data['points']
    O, M, K, C, D, A, B, E = [points[k] for k in ['O', 'M', 'K', 'C', 'D', 'A', 'B', 'E']]
    
    fig = go.Figure()
    
    # Add workspace boundary
    if workspace_boundary is not None and len(workspace_boundary) > 0:
        fig.add_trace(go.Scatter(
            x=workspace_boundary[:, 0],
            y=workspace_boundary[:, 1],
            mode='lines',
            name='Workspace',
            line=dict(color='lightblue', width=2, dash='dash'),
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.2)',
            hoverinfo='skip'
        ))
    
    # Add trajectory path
    if trajectory_path is not None:
        fig.add_trace(go.Scatter(
            x=trajectory_path[:, 0],
            y=trajectory_path[:, 1],
            mode='lines',
            name='Gait Trajectory',
            line=dict(color='blue', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    # Add links
    links = [
        ([O[0], K[0]], [O[1], K[1]], 'Thigh (L1)', 'blue', 4),
        ([K[0], E[0]], [K[1], E[1]], 'Calf (L2)', 'green', 4),
        ([M[0], C[0]], [M[1], C[1]], 'Link L3', 'red', 2.5),
        ([O[0], D[0]], [O[1], D[1]], 'Link L41', 'purple', 2),
        ([D[0], A[0]], [D[1], A[1]], 'Link L42', 'orange', 2),
        ([K[0], B[0]], [K[1], B[1]], 'Link L2a', 'cyan', 2),
        ([C[0], D[0]], [C[1], D[1]], 'Link L51', 'brown', 2),
        ([A[0], B[0]], [A[1], B[1]], 'Link L52', 'pink', 2),
    ]
    
    for x, y, name, color, width in links:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=name,
            line=dict(color=color, width=width),
            hoverinfo='name'
        ))
    
    # Add joints
    joints = [
        (O, 'O (Origin)', 'red', 10),
        (M, 'M (Motor)', 'blue', 10),
        (K, 'K (Knee)', 'green', 10),
        (C, 'C', 'orange', 7),
        (D, 'D', 'purple', 7),
        (A, 'A', 'brown', 7),
        (B, 'B', 'cyan', 7),
    ]
    
    for point, name, color, size in joints:
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]],
            mode='markers',
            name=name,
            marker=dict(size=size, color=color),
            hovertemplate=f'{name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
        ))
    
    # Add foot (draggable/clickable)
    fig.add_trace(go.Scatter(
        x=[E[0]], y=[E[1]],
        mode='markers+text',
        name='Foot (Click to move)',
        marker=dict(size=18, color='darkgreen', symbol='diamond'),
        text=['Foot'],
        textposition='top right',
        hovertemplate='Foot (Click to move)<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Add target position if set
    if target_pos is not None:
        fig.add_trace(go.Scatter(
            x=[target_pos[0]], y=[target_pos[1]],
            mode='markers',
            name='Target',
            marker=dict(size=15, color='red', symbol='x'),
            hovertemplate='Target<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title='Quadruped Leg Kinematics - Click anywhere to move foot',
        xaxis=dict(title='X (mm)', range=[-200, 100], scaleanchor="y", scaleratio=1),
        yaxis=dict(title='Y (mm)', range=[-250, 50]),
        hovermode='closest',
        showlegend=True,
        height=700,
        plot_bgcolor='white',
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
    )
    
    return fig

# ==================== UI ====================

st.title("🦿 Quadruped Robot Leg Kinematics - Full Interactive")
st.markdown("**✨ Click on plot to move foot | Adjust all parameters | Run gait animation | Generate motor table**")

# ==================== Sidebar ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Leg Parameters
    with st.expander("🔧 Leg Parameters", expanded=False):
        st.markdown("**Motor Offsets**")
        col1, col2 = st.columns(2)
        with col1:
            d_x = st.number_input("d_x (mm)", value=float(st.session_state.params['d_x']), step=1.0, format="%.1f")
        with col2:
            d_y = st.number_input("d_y (mm)", value=float(st.session_state.params['d_y']), step=1.0, format="%.1f")
        
        st.markdown("**Link Lengths**")
        L1 = st.number_input("L1 - Thigh (mm)", value=float(st.session_state.params['L1']), step=1.0, format="%.1f")
        L2 = st.number_input("L2 - Calf K-E (mm)", value=float(st.session_state.params['L2']), step=1.0, format="%.1f")
        L2a = st.number_input("L2a - Calf K-B (mm)", value=float(st.session_state.params['L2a']), step=1.0, format="%.1f")
        L3 = st.number_input("L3 - Motor rod (mm)", value=float(st.session_state.params['L3']), step=1.0, format="%.1f")
        
        st.markdown("**Fan Linkage**")
        col1, col2 = st.columns(2)
        with col1:
            L41 = st.number_input("L41 (mm)", value=float(st.session_state.params['L41']), step=1.0, format="%.1f")
            L51 = st.number_input("L51 - CD (mm)", value=float(st.session_state.params['L51']), step=1.0, format="%.2f")
        with col2:
            L42 = st.number_input("L42 (mm)", value=float(st.session_state.params['L42']), step=1.0, format="%.1f")
            L52 = st.number_input("L52 - AB (mm)", value=float(st.session_state.params['L52']), step=1.0, format="%.1f")
        
        phi = st.number_input("φ - Fan angle (deg)", value=float(st.session_state.params['phi']), step=0.01, format="%.2f")
        
        if st.button("✅ Apply Leg Parameters", type="primary", use_container_width=True):
            st.session_state.params = {
                'd_x': d_x, 'd_y': d_y,
                'L1': L1, 'L2': L2, 'L2a': L2a, 'L3': L3,
                'L41': L41, 'L42': L42, 'phi': phi,
                'L51': L51, 'L52': L52
            }
            # Recalculate workspace
            st.session_state.workspace_boundary = calculate_workspace_boundary(st.session_state.params)
            st.success("✓ Parameters applied! Workspace recalculated.")
            st.rerun()
    
    st.markdown("---")
    
    # Gait Parameters
    with st.expander("🚶 Gait Parameters", expanded=True):
        step_length = st.slider("Step Length (mm)", 5, 40, st.session_state.gait_params['step_length'], 1)
        step_height = st.slider("Step Height (mm)", 2, 20, st.session_state.gait_params['step_height'], 1)
        period = st.slider("Period (sec)", 0.5, 2.0, st.session_state.gait_params['period'], 0.1)
        center_x = st.number_input("Center X (mm)", value=float(st.session_state.gait_params['center_x']), step=1.0, format="%.1f")
        center_y = st.number_input("Center Y (mm)", value=float(st.session_state.gait_params['center_y']), step=1.0, format="%.1f")
        
        st.session_state.gait_params = {
            'step_length': step_length,
            'step_height': step_height,
            'period': period,
            'center_x': center_x,
            'center_y': center_y
        }
    
    st.markdown("---")
    
    # Workspace
    if st.button("🔄 Calculate Workspace", use_container_width=True):
        with st.spinner("Calculating workspace boundary..."):
            st.session_state.workspace_boundary = calculate_workspace_boundary(st.session_state.params)
            st.success("✓ Workspace calculated!")
            st.rerun()

# ==================== Main Controls ====================

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("🎮 Motor Control")
    motor1 = st.slider("Thigh Motor (deg)", 60, 120, st.session_state.motor1, 1, 
                       disabled=st.session_state.gait_playing,
                       key="motor1_slider")
    motor2 = st.slider("Calf Motor (deg)", 60, 100, st.session_state.motor2, 1,
                       disabled=st.session_state.gait_playing,
                       key="motor2_slider")
    
    if not st.session_state.gait_playing:
        st.session_state.motor1 = motor1
        st.session_state.motor2 = motor2

with col2:
    st.subheader("🎯 Target Position (IK)")
    col_x, col_y = st.columns(2)
    with col_x:
        target_x = st.number_input("Target X (mm)", value=-10.0, step=1.0, format="%.2f", key="target_x_input")
    with col_y:
        target_y = st.number_input("Target Y (mm)", value=-155.7, step=1.0, format="%.2f", key="target_y_input")
    
    if st.button("🔍 Solve IK", type="primary", use_container_width=True):
        st.session_state.target_pos = np.array([target_x, target_y])
        with st.spinner("Solving IK..."):
            m1, m2, success = inverse_kinematics(
                target_x, target_y,
                (st.session_state.motor1, st.session_state.motor2),
                st.session_state.params
            )
            st.session_state.motor1 = m1
            st.session_state.motor2 = m2
            
            data = calculate_kinematics(m1, m2, st.session_state.params)
            E = data['points']['E']
            error = np.linalg.norm(E - np.array([target_x, target_y]))
            
            if success and error < 1.0:
                st.success(f"✅ IK Solved! Error = {error:.3f} mm")
            else:
                st.warning(f"⚠️ IK approximate. Error = {error:.3f} mm")
        st.rerun()

with col3:
    st.subheader("🔄 Actions")
    if st.button("Reset", use_container_width=True):
        st.session_state.motor1 = 90
        st.session_state.motor2 = 90
        st.session_state.target_pos = None
        st.session_state.trajectory_path = None
        st.session_state.gait_playing = False
        st.rerun()

st.markdown("---")

# ==================== Gait Control ====================
st.subheader("🚶 Gait Animation")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Play Gait", use_container_width=True, disabled=st.session_state.gait_playing):
        st.session_state.gait_playing = True
        st.session_state.gait_time = 0
        st.session_state.trajectory_path = generate_trajectory_path(
            st.session_state.gait_params['step_length'],
            st.session_state.gait_params['step_height'],
            st.session_state.gait_params['period'],
            st.session_state.gait_params['center_x'],
            st.session_state.gait_params['center_y']
        )
        st.rerun()

with col2:
    if st.button("⏸️ Stop Gait", use_container_width=True, disabled=not st.session_state.gait_playing):
        st.session_state.gait_playing = False
        st.rerun()

with col3:
    if st.button("🗑️ Clear Trajectory", use_container_width=True):
        st.session_state.trajectory_path = None
        st.rerun()

with col4:
    if st.button("📊 Generate Table", use_container_width=True):
        if st.session_state.trajectory_path is not None:
            with st.spinner("Generating motor angle table..."):
                period = st.session_state.gait_params['period']
                sample_rate = 20
                num_samples = int(period * sample_rate)
                
                table_data = []
                for i in range(num_samples):
                    t = i / sample_rate
                    tx, ty = foot_trajectory(
                        t,
                        st.session_state.gait_params['step_length'],
                        st.session_state.gait_params['step_height'],
                        st.session_state.gait_params['period'],
                        st.session_state.gait_params['center_x'],
                        st.session_state.gait_params['center_y']
                    )
                    m1, m2, _ = inverse_kinematics(tx, ty, (90, 90), st.session_state.params)
                    table_data.append({
                        'Time (s)': f"{t:.3f}",
                        'Motor 1 (deg)': f"{m1:.2f}",
                        'Motor 2 (deg)': f"{m2:.2f}",
                        'Target X (mm)': f"{tx:.2f}",
                        'Target Y (mm)': f"{ty:.2f}"
                    })
                
                st.session_state.motor_table = table_data
                st.success(f"✓ Generated {len(table_data)} samples")
                st.rerun()
        else:
            st.error("Generate trajectory first (Play Gait)!")

st.markdown("---")

# ==================== Animation Logic ====================
if st.session_state.gait_playing:
    target_x_anim, target_y_anim = foot_trajectory(
        st.session_state.gait_time,
        st.session_state.gait_params['step_length'],
        st.session_state.gait_params['step_height'],
        st.session_state.gait_params['period'],
        st.session_state.gait_params['center_x'],
        st.session_state.gait_params['center_y']
    )
    
    m1, m2, _ = inverse_kinematics(target_x_anim, target_y_anim, 
                                   (st.session_state.motor1, st.session_state.motor2),
                                   st.session_state.params)
    st.session_state.motor1 = m1
    st.session_state.motor2 = m2
    st.session_state.target_pos = np.array([target_x_anim, target_y_anim])
    st.session_state.gait_time += 0.05
    
    time.sleep(0.05)
    st.rerun()

# ==================== Main Visualization ====================
col_plot, col_info = st.columns([3, 1])

with col_plot:
    # Calculate current state
    data = calculate_kinematics(st.session_state.motor1, st.session_state.motor2, st.session_state.params)
    
    # Create Plotly figure
    fig = create_plotly_figure(
        data, 
        st.session_state.workspace_boundary,
        st.session_state.trajectory_path,
        st.session_state.target_pos
    )
    
    # Display and capture clicks
    selected_points = st.plotly_chart(
        fig, 
        use_container_width=True,
        on_select="rerun",
        key="main_plot"
    )
    
    # Handle click events
    if selected_points and selected_points.selection and selected_points.selection.points:
        point = selected_points.selection.points[0]
        if 'x' in point and 'y' in point:
            click_x = point['x']
            click_y = point['y']
            
            st.session_state.target_pos = np.array([click_x, click_y])
            
            # Solve IK
            m1, m2, success = inverse_kinematics(
                click_x, click_y,
                (st.session_state.motor1, st.session_state.motor2),
                st.session_state.params
            )
            
            st.session_state.motor1 = m1
            st.session_state.motor2 = m2
            
            st.rerun()

with col_info:
    st.subheader("📊 System Info")
    
    angles = data['angles']
    E = data['points']['E']
    
    st.metric("Motor 1", f"{st.session_state.motor1:.2f}°")
    st.metric("Motor 2", f"{st.session_state.motor2:.2f}°")
    st.metric("∠KOX", f"{angles['KOX']:.2f}°")
    st.metric("∠OKE", f"{angles['OKE']:.2f}°")
    st.metric("Foot X", f"{E[0]:.2f} mm")
    st.metric("Foot Y", f"{E[1]:.2f} mm")
    
    if st.session_state.gait_playing:
        st.success("🏃 Animation running")
    
    if st.session_state.workspace_boundary is not None:
        st.info(f"Workspace: {len(st.session_state.workspace_boundary)} points")

# ==================== Motor Table ====================
if st.session_state.motor_table:
    st.markdown("---")
    st.subheader("📋 Motor Angle Table")
    
    df = pd.DataFrame(st.session_state.motor_table)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df, use_container_width=True, height=300)
    
    with col2:
        st.download_button(
            label="💾 Download CSV",
            data=df.to_csv(index=False),
            file_name="motor_angles.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        if st.button("🗑️ Clear Table", use_container_width=True):
            st.session_state.motor_table = []
            st.rerun()
        
        st.metric("Samples", len(st.session_state.motor_table))

# ==================== Instructions ====================
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### 🎮 Controls
    
    **1. Adjust Leg Parameters** (Sidebar)
    - Modify d_x, d_y, L1-L52, φ
    - Click "Apply Leg Parameters" to recalculate
    
    **2. Control Motors** (Main area)
    - Use sliders to adjust Motor 1 and Motor 2
    - See leg move in real-time
    
    **3. Inverse Kinematics**
    - Enter Target X, Y coordinates
    - Click "Solve IK" to calculate motor angles
    - **OR** click anywhere on the plot!
    
    **4. Gait Animation**
    - Adjust gait parameters (Sidebar)
    - Click "Play Gait" to start animation
    - Click "Stop" to pause
    - Click "Clear Trajectory" to remove path
    
    **5. Workspace**
    - Click "Calculate Workspace" to see foot reach
    - Blue shaded area shows reachable positions
    
    **6. Generate Motor Table**
    - Play gait first
    - Click "Generate Table"
    - Download as CSV
    
    ### ✨ Interactive Features
    - **Click plot** to move foot to that position
    - **Hover** over points to see coordinates
    - **Zoom** with mouse wheel
    - **Pan** by dragging
    """)

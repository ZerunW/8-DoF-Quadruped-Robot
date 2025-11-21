import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# ==================== Page Config ====================
st.set_page_config(
    page_title="Quadruped Robot - Interactive with Plotly",
    page_icon="🦿",
    layout="wide"
)

# ==================== Session State ====================
if 'motor1' not in st.session_state:
    st.session_state.motor1 = 90
if 'motor2' not in st.session_state:
    st.session_state.motor2 = 90
if 'params' not in st.session_state:
    st.session_state.params = {
        'd_x': -20, 'd_y': -20,
        'L1': 110, 'L2': 110, 'L2a': 30, 'L3': 24,
        'L41': 24, 'L42': 30, 'phi': 90.05,
        'L51': 28.2, 'L52': 110
    }

# ==================== Kinematics Functions ====================

def calculate_kinematics(motor_angle1, motor_angle2, params):
    """Forward kinematics"""
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

# ==================== Plotly Plotting ====================

def create_plotly_figure(data):
    """Create interactive Plotly figure"""
    points = data['points']
    O, M, K, C, D, A, B, E = [points[k] for k in ['O', 'M', 'K', 'C', 'D', 'A', 'B', 'E']]
    
    fig = go.Figure()
    
    # Add links as lines
    links = [
        ([O[0], K[0]], [O[1], K[1]], 'Thigh (L1)', 'blue', 3),
        ([K[0], E[0]], [K[1], E[1]], 'Calf (L2)', 'green', 3),
        ([M[0], C[0]], [M[1], C[1]], 'Link L3', 'red', 2),
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
    
    # Add joints as markers
    joints = [
        (O, 'O (Origin)', 'red', 8),
        (M, 'M (Motor)', 'blue', 8),
        (K, 'K (Knee)', 'green', 8),
        (C, 'C', 'orange', 6),
        (D, 'D', 'purple', 6),
        (A, 'A', 'brown', 6),
        (B, 'B', 'cyan', 6),
    ]
    
    for point, name, color, size in joints:
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]],
            mode='markers',
            name=name,
            marker=dict(size=size, color=color),
            hovertemplate=f'{name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
        ))
    
    # Add foot as draggable point
    fig.add_trace(go.Scatter(
        x=[E[0]], y=[E[1]],
        mode='markers+text',
        name='Foot (Draggable)',
        marker=dict(size=15, color='darkgreen', symbol='circle'),
        text=['Foot'],
        textposition='top right',
        hovertemplate='Foot<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Click and drag to move!<extra></extra>',
        # This makes it "draggable" - user can see coordinates change
    ))
    
    # Layout
    fig.update_layout(
        title='Quadruped Leg Kinematics - Click on the plot to set target position',
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

st.title("🦿 Quadruped Robot Leg Kinematics - Interactive Version")
st.markdown("**✨ Click anywhere on the plot to move the foot to that position!**")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    motor1 = st.slider("Thigh Motor (deg)", 60, 120, st.session_state.motor1, 1)
    motor2 = st.slider("Calf Motor (deg)", 60, 100, st.session_state.motor2, 1)
    
    st.session_state.motor1 = motor1
    st.session_state.motor2 = motor2
    
    if st.button("Reset", use_container_width=True):
        st.session_state.motor1 = 90
        st.session_state.motor2 = 90
        st.rerun()

# Calculate current state
data = calculate_kinematics(st.session_state.motor1, st.session_state.motor2, st.session_state.params)
E = data['points']['E']
angles = data['angles']

# Display info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Motor 1", f"{st.session_state.motor1:.1f}°")
with col2:
    st.metric("Motor 2", f"{st.session_state.motor2:.1f}°")
with col3:
    st.metric("Foot X", f"{E[0]:.2f} mm")
with col4:
    st.metric("Foot Y", f"{E[1]:.2f} mm")

# Create and display Plotly figure
fig = create_plotly_figure(data)

# Capture click events
selected_points = st.plotly_chart(
    fig, 
    use_container_width=True,
    on_select="rerun",  # This enables click interaction
    key="plotly_chart"
)

# Handle click events
if selected_points and selected_points.selection and selected_points.selection.points:
    # User clicked on the plot
    point = selected_points.selection.points[0]
    if 'x' in point and 'y' in point:
        target_x = point['x']
        target_y = point['y']
        
        st.info(f"🎯 Clicked at X: {target_x:.2f}, Y: {target_y:.2f} - Solving IK...")
        
        # Solve IK for clicked position
        m1, m2, success = inverse_kinematics(
            target_x, target_y,
            (st.session_state.motor1, st.session_state.motor2),
            st.session_state.params
        )
        
        # Update motors
        st.session_state.motor1 = m1
        st.session_state.motor2 = m2
        
        # Show result
        new_data = calculate_kinematics(m1, m2, st.session_state.params)
        new_E = new_data['points']['E']
        error = np.linalg.norm(new_E - np.array([target_x, target_y]))
        
        if success and error < 1.0:
            st.success(f"IK Solved! Error = {error:.3f} mm")
        else:
            st.warning(f"IK approximate. Error = {error:.3f} mm")
        
        st.rerun()

# Instructions
with st.expander("How to Use", expanded=False):
    st.markdown("""
    ### Interactive Controls
    
    **Method 1: Click on Plot**
    - Simply click anywhere on the plot
    - The foot will move to that position
    - Motor angles update automatically
    
    **Method 2: Use Sliders**
    - Adjust Motor 1 and Motor 2 sliders in the sidebar
    - See the leg configuration change in real-time
    
    **Method 3: Hover to See Coordinates**
    - Hover over any point to see its coordinates
    - This helps you plan where to click
    
    ### Why This Works
    - Plotly charts are interactive in the browser
    - Click events are captured by Streamlit
    - IK solver runs automatically when you click
    """)


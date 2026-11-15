"""3D model for Foxglove or RViz: robot_state_publisher + joint_state_from_servo.

    ros2 launch robot_dog_description description.launch.py               # robot or fake driver
    ros2 launch robot_dog_description description.launch.py use_sim:=true # with robot_dog_sim

With use_sim:=false, joint_state_from_servo publishes /joint_states and the TF odom -> base_link.
With use_sim:=true the sim publishes them, and unless estimate:=false a second model with the
frame prefix "estimate/" shows the commanded legs.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from robot_dog_description.model import robot_description


def _bool(context, name):
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f'launch argument {name}:={value!r} is not a boolean')


def _nodes(context):
    use_sim = _bool(context, 'use_sim')
    estimate = _bool(context, 'estimate')
    pose_source = LaunchConfiguration('pose_source').perform(context).strip()
    if not pose_source:
        pose_source = 'sim_ground_truth' if use_sim else 'imu_dead_reckoning'
    urdf = ParameterValue(robot_description(), value_type=str)
    actions = [
        LogInfo(msg=f'robot_dog_description: use_sim={use_sim} pose_source={pose_source}'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher', output='screen',
             ros_arguments=['--log-level', 'warn'],
             parameters=[{'robot_description': urdf, 'publish_frequency': 50.0}]),
    ]
    if not use_sim:
        actions.append(Node(
            package='robot_dog_description', executable='joint_state_from_servo_node',
            name='joint_state_from_servo', output='screen', emulate_tty=True,
            parameters=[{'pose_source': pose_source}]))
    elif estimate:
        actions.append(Node(
            package='robot_dog_description', executable='joint_state_from_servo_node',
            name='joint_state_from_servo', namespace='estimate', output='screen',
            emulate_tty=True,
            parameters=[{'pose_source': pose_source, 'frame_prefix': 'estimate/'}]))
        actions.append(Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher', namespace='estimate', output='screen',
            ros_arguments=['--log-level', 'warn'],
            parameters=[{'robot_description': urdf, 'frame_prefix': 'estimate/',
                         'publish_frequency': 50.0}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='false',
                              description='true: robot_dog_sim publishes /joint_states + TF'),
        DeclareLaunchArgument('pose_source', default_value='',
                              description='imu_dead_reckoning | sim_ground_truth '
                                          '(default: sim_ground_truth if use_sim)'),
        DeclareLaunchArgument('estimate', default_value='true',
                              description='use_sim only: also show the servo-state model '
                                          '(frame prefix estimate/)'),
        OpaqueFunction(function=_nodes),
    ])

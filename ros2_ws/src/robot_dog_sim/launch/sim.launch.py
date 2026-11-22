"""MuJoCo simulation with the real servo_driver (fake_hw) and gait_controller.

    pixi run launch_sim                       # headless; Foxglove ws://localhost:8765
    pixi run launch_sim_viewer                # with the MuJoCo viewer window
    ros2 launch robot_dog_sim sim.launch.py ground_tilt_pitch_deg:=3 payload_mass_kg:=0.2

Nodes: sim, servo_driver, gait_controller, the robot_dog_description 3D model and
foxglove_bridge. Disturb it with `ros2 service call /sim/push std_srvs/srv/Trigger`, reset it
with `ros2 service call /sim/reset std_srvs/srv/Trigger`, or tilt the ground with
`ros2 param set /sim ground_tilt_pitch_deg 3.0`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node

from robot_dog_bringup.launch_helpers import (stand_pose, bool_arg, description_launch, float_arg,
                                              foxglove_node, param_files, robot_config,
                                              servo_limits)


def _nodes(context):
    files = param_files()
    return [
        LogInfo(msg=f'robot_dog params: {files}  (MuJoCo sim)'),
        Node(package='robot_dog_sim', executable='sim_node', name='sim', output='screen',
             emulate_tty=True,
             parameters=[{'viewer': bool_arg(context, 'viewer'),
                          'ground_tilt_roll_deg': float_arg(context, 'ground_tilt_roll_deg'),
                          'ground_tilt_pitch_deg': float_arg(context, 'ground_tilt_pitch_deg'),
                          'payload_mass_kg': float_arg(context, 'payload_mass_kg')}]),
        Node(package='robot_dog_hw', executable='servo_driver_node', name='servo_driver',
             output='screen', emulate_tty=True, parameters=files + [stand_pose(), {'fake_hw': True}]),
        Node(package='robot_dog_control', executable='gait_controller_node',
             name='gait_controller', output='screen', emulate_tty=True,
             parameters=files + [servo_limits(), {'level_enable': bool_arg(context, 'level')}]),
        description_launch(use_sim=True, pose_source='sim_ground_truth',
                           estimate=bool_arg(context, 'estimate')),
    ] + ([foxglove_node()] if bool_arg(context, 'foxglove') else [])


def generate_launch_description():
    level_default = str(robot_config().get('gait_controller', {})
                        .get('level_enable', True)).lower()
    return LaunchDescription([
        DeclareLaunchArgument('viewer', default_value='false',
                              description='MuJoCo passive viewer window (needs a display)'),
        DeclareLaunchArgument('foxglove', default_value='true',
                              description='start foxglove_bridge (ws://localhost:8765)'),
        DeclareLaunchArgument('level', default_value=level_default,
                              description='IMU leveling (default: level_enable in robot_dog.yaml)'),
        DeclareLaunchArgument('estimate', default_value='true',
                              description='also show the /servo_state model (prefix estimate/)'),
        DeclareLaunchArgument('ground_tilt_roll_deg', default_value='0.0',
                              description='floor slope, + = right side lower'),
        DeclareLaunchArgument('ground_tilt_pitch_deg', default_value='0.0',
                              description='floor slope, + = nose lower'),
        DeclareLaunchArgument('payload_mass_kg', default_value='0.0',
                              description='extra mass on the body (battery, ...)'),
        OpaqueFunction(function=_nodes),
    ])

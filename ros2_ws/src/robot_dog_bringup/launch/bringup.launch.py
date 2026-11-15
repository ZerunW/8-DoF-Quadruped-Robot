"""Full robot bringup: the dog stands up and levels.

    pixi run launch_bringup
    ros2 launch robot_dog_bringup bringup.launch.py teleop:=true

Starts servo_driver, the BNO055 driver with its supervisor, gait_controller and the 3D
model. The controller waits for /servo_state, then blends from the driver's power-on pose
to its stand pose. The Foxglove bridge starts here only with foxglove:=true;
``pixi run launch_foxglove`` runs it on its own.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node

from robot_dog_bringup.launch_helpers import (bool_arg, config_file, description_launch,
                                              foxglove_node, param_files, robot_config,
                                              servo_limits, stand_pose, teleop_nodes)


def _nodes(context):
    files = param_files()
    actions = [
        LogInfo(msg=f'robot_dog params: {files}'),
        Node(package='robot_dog_hw', executable='servo_driver_node', name='servo_driver',
             output='screen', emulate_tty=True, parameters=files + [stand_pose()]),
        # Respawned when imu_supervisor kills it.
        Node(package='bno055', executable='bno055', name='bno055', output='screen',
             emulate_tty=True, parameters=[config_file('robot_dog.yaml')],
             respawn=True, respawn_delay=2.0),
        Node(package='robot_dog_hw', executable='imu_supervisor_node', name='imu_supervisor',
             output='screen', emulate_tty=True),
        Node(package='robot_dog_control', executable='gait_controller_node',
             name='gait_controller', output='screen', emulate_tty=True,
             parameters=files + [servo_limits(), {'level_enable': bool_arg(context, 'level')}]),
    ]
    if bool_arg(context, 'viz'):
        actions.append(description_launch(use_sim=False, pose_source='imu_dead_reckoning'))
    if bool_arg(context, 'foxglove'):
        actions.append(foxglove_node())
    if bool_arg(context, 'teleop'):
        actions += teleop_nodes()
    return actions


def generate_launch_description():
    level_default = str(robot_config().get('gait_controller', {})
                        .get('level_enable', True)).lower()
    return LaunchDescription([
        DeclareLaunchArgument('viz', default_value='true',
                              description='publish the 3D model (/joint_states, TF)'),
        DeclareLaunchArgument('foxglove', default_value='false',
                              description='start foxglove_bridge in this launch'),
        DeclareLaunchArgument('teleop', default_value='false',
                              description='PS5 gamepad teleoperation (joy_node + gamepad_teleop)'),
        DeclareLaunchArgument('level', default_value=level_default,
                              description='IMU leveling (default: level_enable in robot_dog.yaml)'),
        OpaqueFunction(function=_nodes),
    ])

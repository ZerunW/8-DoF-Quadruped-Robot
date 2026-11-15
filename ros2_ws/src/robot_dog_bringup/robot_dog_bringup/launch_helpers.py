"""Helpers shared by the launch files."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'robot_dog_bringup'

#: Asset allowlist of foxglove_bridge: its defaults plus the robot_dog_description meshes.
FOXGLOVE_ASSET_ALLOWLIST = [
    r'^package://(?:[-\w%]+/)*[-\w%.]+\.'
    r'(?:dae|fbx|glb|gltf|jpeg|jpg|mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$',
    r'^package://robot_dog_description/.*',
]


def config_file(name: str) -> str:
    return os.path.join(get_package_share_directory(PKG), 'config', name)


def param_files():
    """Parameter files of every node: robot_dog.yaml holds all sections."""
    return [config_file('robot_dog.yaml')]


def robot_config():
    from robot_dog_control.config import load_robot_config
    return load_robot_config(Path(config_file('robot_dog.yaml')))


def stand_pose():
    """{'stand_angles': [...]}: the controller's IK stand pose in servo degrees, for
    servo_driver."""
    from robot_dog_control.config import stand_servo_angles
    return {'stand_angles': stand_servo_angles(robot_config())}


def servo_limits():
    """servo_min / servo_max of servo_driver, for the controller's envelope check."""
    drv = robot_config().get('servo_driver', {})
    return {'servo_min': [float(v) for v in drv['servo_min']],
            'servo_max': [float(v) for v in drv['servo_max']]}


def bool_arg(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f'launch argument {name}:={value!r} is not a boolean')


def float_arg(context, name: str) -> float:
    return float(LaunchConfiguration(name).perform(context))


def joy_node() -> Node:
    """joy_node (SDL2) reading the first gamepad, configured by the joy_node section of
    robot_dog.yaml."""
    return Node(package='joy', executable='joy_node', name='joy_node', output='screen',
                emulate_tty=True, parameters=param_files())


def teleop_nodes():
    """joy_node and gamepad_teleop, configured by robot_dog.yaml."""
    return [joy_node(),
            Node(package='robot_dog_control', executable='gamepad_teleop', name='gamepad_teleop',
                 output='screen', emulate_tty=True, parameters=param_files())]


def foxglove_node() -> Node:
    """foxglove_bridge on ws://<host>:8765 with FOXGLOVE_ASSET_ALLOWLIST."""
    return Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
                output='screen', ros_arguments=['--log-level', 'warn'],
                parameters=[{'max_qos_depth': 100,
                             'asset_uri_allowlist': FOXGLOVE_ASSET_ALLOWLIST}])


def description_launch(use_sim: bool, pose_source: str = '', estimate: bool = True):
    """Include robot_dog_description/launch/description.launch.py."""
    path = os.path.join(get_package_share_directory('robot_dog_description'), 'launch',
                        'description.launch.py')
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments={'use_sim': str(use_sim).lower(), 'pose_source': pose_source,
                          'estimate': str(estimate).lower()}.items())

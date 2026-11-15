"""servo_driver alone, for servo identification, offset tuning and leg_test sessions.

    pixi run launch_driver                                         # real PCA9685
    ros2 launch robot_dog_bringup driver.launch.py fake_hw:=true   # dry run
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node

from robot_dog_bringup.launch_helpers import bool_arg, param_files, stand_pose


def _nodes(context):
    files = param_files()
    return [
        LogInfo(msg=f'robot_dog params: {files}'),
        Node(package='robot_dog_hw', executable='servo_driver_node', name='servo_driver',
             output='screen', emulate_tty=True,
             parameters=files + [stand_pose(), {'fake_hw': bool_arg(context, 'fake_hw')}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('fake_hw', default_value='false',
                              description='use a fake PCA9685 (no I2C)'),
        OpaqueFunction(function=_nodes),
    ])

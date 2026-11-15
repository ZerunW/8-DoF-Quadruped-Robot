"""PS5 DualSense teleoperation: joy_node and gamepad_teleop; needs gait_controller running.

    pixi run pair_gamepad                                           # pair the gamepad first
    pixi run launch_teleop                                          # next to launch_bringup
    ros2 launch robot_dog_bringup bringup.launch.py teleop:=true    # same, inside bringup

The dog moves only while L1 is held; docs/hardware.md lists the controls.
"""

from launch import LaunchDescription

from robot_dog_bringup.launch_helpers import teleop_nodes


def generate_launch_description():
    return LaunchDescription(teleop_nodes())

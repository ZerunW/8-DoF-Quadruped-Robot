"""Foxglove bridge on its own (ws://<host>:8765), separate from the control stack.

    pixi run launch_foxglove        # on the Pi, then connect Foxglove to ws://<pi-host>:8765

The bridge serves the URDF meshes of robot_dog_description through its asset allowlist.
"""

from launch import LaunchDescription

from robot_dog_bringup.launch_helpers import foxglove_node


def generate_launch_description():
    return LaunchDescription([foxglove_node()])

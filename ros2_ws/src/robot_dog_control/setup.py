from setuptools import find_packages, setup

package_name = 'robot_dog_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zerun Wang',
    maintainer_email='zerunwang@gatech.edu',
    description='Robot dog IK gait controller with IMU leveling and heading hold',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gait_controller_node = robot_dog_control.gait_controller_node:main',
            'gamepad_teleop = robot_dog_control.gamepad_teleop_node:main',
            'leg_test = robot_dog_control.tools.leg_test:main',
            'calib_offsets = robot_dog_control.tools.calib_offsets:main',
            'walk_test = robot_dog_control.tools.walk_test:main',
        ],
    },
)

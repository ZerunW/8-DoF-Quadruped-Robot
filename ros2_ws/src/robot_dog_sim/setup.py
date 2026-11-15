import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_dog_sim'


def files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), files('launch/*.launch.py')),
        (os.path.join('share', package_name, 'mjcf'), files('mjcf/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zerun Wang',
    maintainer_email='zerunwang@gatech.edu',
    description='MuJoCo physics simulation of the robot dog (closed-loop legs, hobby-servo '
                'model, IMU, ground truth)',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'sim_node = robot_dog_sim.sim_node:main',
            'sim_push = robot_dog_sim.tools.sim_push:main',
            'sim_reset = robot_dog_sim.tools.sim_reset:main',
        ],
    },
)

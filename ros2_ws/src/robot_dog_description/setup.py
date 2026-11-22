import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_dog_description'


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
        (os.path.join('share', package_name, 'urdf'), files('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes'), files('meshes/*.stl')),
        (os.path.join('share', package_name, 'config'), files('config/*.yaml')),
        (os.path.join('share', package_name, 'foxglove'), files('foxglove/*.json')),
        # front/rear unsafe-pair sets
        (os.path.join('share', package_name, 'doc'), files('doc/pair_sweep_*_final.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zerun Wang',
    maintainer_email='zerunwang@gatech.edu',
    description='Robot dog URDF (CAD meshes, open-tree 5-bar legs) and joint states from '
                'the servo state',
    license='MIT',
    entry_points={
        'console_scripts': [
            'joint_state_from_servo_node = '
            'robot_dog_description.joint_state_from_servo_node:main',
        ],
    },
)

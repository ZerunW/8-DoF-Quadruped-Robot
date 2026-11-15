from setuptools import find_packages, setup

package_name = 'robot_dog_hw'

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
    description='Robot dog hardware I/O (PCA9685 servo driver, IMU supervisor) and tools',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'servo_driver_node = robot_dog_hw.servo_driver_node:main',
            'imu_supervisor_node = robot_dog_hw.imu_supervisor_node:main',
            'servo_tune = robot_dog_hw.tools.servo_tune:main',
            'relax = robot_dog_hw.tools.relax:main',
            'stand = robot_dog_hw.tools.stand:main',
            'imu_check = robot_dog_hw.tools.imu_check:main',
        ],
    },
)

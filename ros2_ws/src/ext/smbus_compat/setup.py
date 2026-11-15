from setuptools import setup

package_name = 'smbus_compat'

setup(
    name=package_name,
    version='0.1.0',
    py_modules=['smbus'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zerun Wang',
    maintainer_email='zerunwang@gatech.edu',
    description='smbus -> smbus2 shim for ros-humble-bno055',
    license='MIT',
)

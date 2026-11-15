"""Hardware I/O for the robot dog: PCA9685 servo driver, IMU supervisor, tools.

Only `servo_driver_node` writes to the PCA9685, always through `servo_map.ServoLimiter`.
Everything else publishes `/servo_cmd` or uses the driver's services and parameters.
"""

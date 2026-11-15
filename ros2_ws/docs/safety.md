# Safety

## Servo driver

- Only `servo_driver` writes to the PCA9685, through `ServoLimiter`: clamp to
  `[servo_min, servo_max]`, add the offset, clip the pulse to 500–2500 µs.
- The servo configuration is validated at startup. Offsets are limited to ±40° and live
  offset changes to 10° per step.
- Commands are rate limited to `max_servo_speed_dps` (400°/s). NaN and infinite entries keep
  the previous target.
- With no `/servo_cmd` for `cmd_timeout_s`, the driver holds the last pose.
- On shutdown, the driver moves to the stand pose and leaves the outputs on. `relax` is the
  only way to turn them off.
- A PCA9685 that reset after a brown-out is reinitialized and the current pose rewritten.

## Controller

- IK stays inside the joint region from the CAD.
- At startup the controller checks every pose it can plan against IK, servo limits and the
  front/rear leg collision set, and refuses to walk if something does not fit.
- Every tick, each side's (rear, front) joint pair is looked up in the collision set. A
  marked pair holds both legs at their last safe output.
- Multi-servo moves such as the startup blend are straight lines in joint space.
- Scheduled motion stays at or below 5.6° of servo travel per 20 ms tick.

## Network

The Pixi environment sets `ROS_LOCALHOST_ONLY=1`, so a simulation on the laptop cannot
reach the robot's `/servo_cmd`.

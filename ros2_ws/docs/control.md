# Control

## Nodes

```
/cmd_vel /cmd_step /cmd_pose /cmd_heading     /bno055/imu
                 \                                 |
                  +------> gait_controller <-------+
                                |  /servo_cmd (50 Hz)
                                v
                          servo_driver --> PCA9685 --> 8 servos
                                |  /servo_state
                                v
                   joint_state_from_servo --> /joint_states, TF
```

`gait_controller` wraps `GaitControllerCore` (`controller.py`), the controller without ROS.
The tests drive the core directly.

## Conventions

- Leg order is `lf, rf, lb, rb`.
- Body frame is REP-103: x forward, y left, z up. **+pitch is nose down, +roll is right
  side down, +yaw turns left.**
- Foot targets are `(fx, fz)` in mm: `fx` forward of the hip axis, `fz` up from it.
- Motor angle to servo angle: `s = m + 90` on the left, `s = 360 - (m + 90)` on the right.

## Pipeline per tick

1. **Gait** (`gait.py`) plans each foot in the support frame. Gaits are presets in
   `gait_presets` (phase offsets, duty, period, step height, Bézier swing), and the speed
   schedule picks the preset from the body speed.
2. **Pose layer** (`pose.py`) applies body height, pitch, roll and fore/aft shift as one rigid
   transform, clamped to the pose limits.
3. **Leveling** adds `PID(level_target + commanded_attitude - imu_attitude)` to the commanded
   pitch and roll.
4. **Heading hold** steers the gait with `w = Kp·wrap(yaw_ref - yaw) - Kd·gyro_z` while no
   turn is commanded, limited to `heading_max_rps`.
5. **IK** (`kinematics.py`) solves both motor angles inside the joint region.
6. **Pair guard** (`unsafe_pairs.py`) holds a side's legs if their joint pair is in the
   collision set.

At startup the controller validates every pose it can plan (see [safety.md](safety.md)).

## Behavior API

```python
from robot_dog_control.api import RobotDog
dog = RobotDog()
dog.walk(height=155, heading=0, speed=0.08, duration=5)
dog.walk(speed=-0.05, distance=0.3)
dog.step_in_place(duration=4, heading=90)
dog.set_height(130)
dog.set_lean(pitch_deg=10)
dog.close()
```

Headings count from where the dog faced when the client connected. A distance walk
integrates the gait's planned speed.

## Topics

| Topic | Type | |
|---|---|---|
| `/cmd_vel` | Twist | `linear.x` m/s, `angular.z` rad/s |
| `/cmd_step` | Bool | step in place while true |
| `/cmd_pose` | Twist | `linear.z` height (m), `angular.y` pitch, `angular.x` roll (rad) |
| `/cmd_heading` | Float32 | heading target (rad) |
| `/servo_cmd`, `/servo_state` | JointState | servo angles (deg), commanded and written |
| `/servo_driver/clamped` | Bool | the latest command needed clamping |
| `/bno055/imu` | Imu | body orientation |
| `/gait/debug` | Float64MultiArray | layout in `controller.DEBUG_FIELDS` |

## Services

| Service | Type |
|---|---|
| `/gait_controller/stand`, `/rest`, `/estop`, `/zero_yaw` | Trigger |
| `/gait_controller/level_enable` | SetBool |
| `/servo_driver/stand`, `/servo_driver/relax` | Trigger |

## Configuration

`src/robot_dog_bringup/config/robot_dog.yaml` is the only configuration file. Most
`gait_controller` parameters can be changed live with `ros2 param set`, and changes that
would break the envelope check are rejected.

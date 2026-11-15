# 8-DoF Quadruped Robot

A 3D-printed 8-DoF quadruped robot that walks dynamically using IMU-based balance control,
inverse kinematics and coordinated gait generation, with RL training for terrain adaptation
planned.

![Stand pose](ros2_ws/src/robot_dog_description/doc/stand_pose.png)

## The robot

- **Legs**: four planar five-bar linkages, each driven by two Feetech FT6335 360° servos at
  the hip. The knees point backward, and the feet are TPU spheres.
- **Electronics**: Raspberry Pi 5, PCA9685 servo driver and BNO055 IMU on I2C.
- **Size**: hips 183 mm apart front to back and 176 mm side to side, 155 mm standing hip
  height, about 1.4 kg without the battery.

## What it can do

- Trot forward and backward, and turn in place or while walking
- Hold its heading with the IMU
- Keep the body level on slopes
- Change body height and lean while standing or walking
- Be driven by a PS5 gamepad or scripted through a Python API
- Run the same software in a MuJoCo simulation

## Results

### On the robot

| | |
|---|---|
| Gait | trot: duty 0.7, period 0.6 s, 20 mm step height, stride up to 65 mm |
| Walking speed | up to 0.12 m/s forward and backward (software speed cap) |
| Heading hold | 5 s walk at 0.12 m/s: −1.0° heading error forward, +0.5° backward |
| Body motion while trotting | roll 0.9–1.5° RMS, pitch peak 2.6° |
| Standing | body held level within the 1° leveling deadband |
| Body height | 70 mm (knees on the ground) to 193 mm standing, 116–193 mm while walking |
| Servo limits | no command clamped by the driver in any walk |

### In simulation (MuJoCo, same controller and driver)

| | |
|---|---|
| Distance covered at 0.12 m/s, 155 mm | about 88 % of the commanded distance forward, 98 % backward |
| Walking at 116 / 193 mm height | 87–104 % of the commanded distance |
| 4° slope | body leveled to within 1° in under 5 s, stays level while walking up it |
| 4 N side push for 0.2 s | 1.8° peak roll, back within 1° in 0.2 s |
| Body pose | push-ups 175 ↔ 130 mm, leans of 10° pitch and 8° roll |

### Limitations

- The robot has no position sensor. Distance walks use the gait's own dead reckoning, so
  foot slip shortens them, and stepping in place can drift slowly.
- Turning in place to a target heading is limited to 0.1 rad/s. Turning by speed command
  (gamepad, `/cmd_vel`) goes up to 0.6 rad/s.
- Running (bound) and crawl gaits are configured but not usable yet, and jumping is not
  implemented.
- The full joint range computed from the CAD, including the deepest folded poses, has not
  been verified on the robot yet.

## Repository

| Folder | |
|---|---|
| [`ros2_ws/`](ros2_ws) | ROS 2 Humble control software and MuJoCo simulation |
| [`cad/`](cad) | mechanical design (coming soon) |
| [`pcb/`](pcb) | electronics (coming soon) |

## Software

Hardware drivers, kinematics, gait generation, balance control, safety layers, calibration
tools and the simulation are all in [`ros2_ws/`](ros2_ws). The setup uses
[Pixi](https://pixi.sh), so no system ROS install is needed.

```bash
cd ros2_ws
pixi install && pixi run build
pixi run launch_sim          # simulation on a laptop
```

To run on the robot, see [ros2_ws/README.md](ros2_ws/README.md#running-on-the-robot) and
the [bring-up guide](ros2_ws/docs/bringup.md).

## License

[MIT](LICENSE)

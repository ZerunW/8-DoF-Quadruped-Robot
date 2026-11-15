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

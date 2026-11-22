<div align="center">
    <img src="ros2_ws/src/robot_dog_description/doc/stand_pose.png" alt="Robot standing" width="480">
    <h1>8-DoF Quadruped Robot</h1>
</div>

<p align="center">
    <em>A 3D-printed quadruped that trots with IMU leveling, heading hold and a MuJoCo twin.</em>
</p>

<p align="center">
    <a href="https://github.com/ZerunW/8-DoF-Quadruped-Robot/actions/workflows/lint.yml"><img src="https://img.shields.io/github/actions/workflow/status/ZerunW/8-DoF-Quadruped-Robot/lint.yml?branch=main&style=flat-square&label=lint" alt="Lint status" /></a>
    <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?style=flat-square" alt="ROS 2 Humble" />
    <img src="https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white" alt="Raspberry Pi 5" />
    <img src="https://img.shields.io/badge/MuJoCo-3.16-0A84FF?style=flat-square" alt="MuJoCo" />
    <img src="https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-A855F7?style=flat-square" alt="License: MIT" /></a>
</p>

## What is this?

A quadruped robot with eight actuated joints: each leg is a planar five-bar linkage driven by
two hip servos. The repository holds the ROS 2 software that runs it (servo driver, forward
and inverse kinematics, gait generator, IMU balance controller, safety layers, gamepad and
Python control), a MuJoCo simulation built from the same CAD, and the mechanical and
electronics folders. The controller and servo driver run unchanged on the robot's Raspberry
Pi 5 and in simulation.

## Why?

Eight servos and a five-bar leg are cheap to build and print, but they leave little margin:
the front and rear legs of a side can collide, and a bad command can drive a linkage past its
limits. The software therefore checks every pose it can plan at startup, guards leg pairs on
every tick, and clamps and rate-limits at the servo driver. Simulation with the real driver
in the loop lets you try gaits and disturbances before the robot touches the ground. An RL
policy is planned on top of this (see [Roadmap](#roadmap)).

## Quick start

Simulation on a laptop (linux-64) needs only [Pixi](https://pixi.sh):

```bash
cd ros2_ws
pixi install && pixi run build
pixi run launch_sim            # headless; Foxglove bridge on ws://localhost:8765
```

In a second shell, walk for 5 s with the behavior API, then push the body sideways:

```bash
pixi run python -c "from robot_dog_control.api import RobotDog; d = RobotDog(); d.walk(speed=0.08, duration=5); d.close()"
pixi run ros2 service call /sim/push std_srvs/srv/Trigger    # 3 N to the left for 0.2 s
```

Use `pixi run launch_sim_viewer` for the MuJoCo window. For the robot, see
[Running on the robot](#running-on-the-robot).

## Features

- **Kinematics** — exact forward kinematics of the five-bar leg and Gauss-Newton IK, restricted to a joint region computed from the CAD
- **Gaits as data** — phase offsets, duty, period and Bezier swing per preset; the commanded speed selects the gait
- **Balance** — IMU roll/pitch leveling on slopes and IMU heading hold while walking
- **Body pose** — height, pitch and roll as one rigid transform, clamped to validated limits
- **Safety** — startup check of every plannable pose, per-tick front/rear leg collision guard, servo driver with hard clamp, rate limiter and watchdog
- **Control interfaces** — Python behavior API, PS5 DualSense teleoperation, plain ROS topics
- **Simulation** — MuJoCo model of the full CAD with adjustable ground tilt, payload, servo gains and IMU noise
- **Tooling** — Pixi workspace on `robostack-humble` (no system ROS), and a deploy terminal that syncs and runs tasks on the Pi

## Hardware

| Part | Detail |
|---|---|
| Computer | Raspberry Pi 5, Ubuntu 24.04 |
| Servos | 8 x Feetech FT6335, 360 degrees, PWM 500-2500 us |
| Servo driver | PCA9685 on the servo shield in [`pcb/`](pcb), I2C bus 1, address 0x40, 50 Hz |
| IMU | BNO055, I2C bus 1, address 0x28, IMUPLUS mode |
| Gamepad (optional) | PS5 DualSense over Bluetooth |
| Feet | TPU spheres, 17.5 mm radius |

The robot weighs about 1.4 kg without the battery. The PCA9685 and BNO055 are powered from the
servo rail, so they only appear on I2C while motor power is on. Wiring, servo channels and IMU
mount are in [ros2_ws/docs/hardware.md](ros2_ws/docs/hardware.md). STEP files of the printed
parts are in [`cad/`](cad), and the servo shield (schematic, layout, BOM) is in [`pcb/`](pcb).
The meshes used by the software are in `ros2_ws/src/robot_dog_description/meshes/`.

## Results

### On the robot

| | |
|---|---|
| Gait | trot: duty 0.7, period 0.6 s, 20 mm step height, stride up to 65 mm |
| Walking speed | 0.12 m/s forward and backward |
| Heading hold | 5 s walk at 0.12 m/s: −1.0° heading error forward, +0.5° backward |
| Body motion while trotting | roll 0.9–1.5° RMS, pitch peak 2.6° |
| Standing | body held level to within 1° |
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

## Running on the robot

The laptop holds the workspace. The deploy terminal syncs it to the Pi with rsync and runs
Pixi tasks there over SSH. On the Pi, install Pixi (linux-aarch64), enable I2C and add your
user to the `i2c` group.

```text
cd ros2_ws && pixi run deploy
robotdog> config                    # host, user, deploy directory, SSH key
robotdog> pixi run build
robotdog> pixi run launch_driver    # servo driver only, holds the stand pose
robotdog> pixi run launch_bringup   # full stack: stands up and levels
```

Calibrate servo offsets and the IMU level on first power-up with `servo_tune`, `leg_test`,
`imu_check` and `calib_offsets`. The procedure is in
[ros2_ws/docs/bringup.md](ros2_ws/docs/bringup.md). Drive it with a gamepad
(`pixi run launch_bringup teleop:=true`; L1 is the deadman, Cross is emergency stop) or from
Python as below. Stop with `pixi run stand`; turn the servos off with `pixi run relax`.

Control from Python:

```python
from robot_dog_control.api import RobotDog

dog = RobotDog()
dog.walk(height=155, heading=0, speed=0.08, duration=5)   # mm, deg, m/s, s
dog.set_lean(pitch_deg=10, roll_deg=0)
dog.close()
```

## How it works

```text
BNO055 IMU ----------------------------+
 /bno055/imu                           v
gamepad / API / topics -->  gait_controller (50 Hz)
 /cmd_vel /cmd_step          1. gait planner: foot paths per leg
 /cmd_pose /cmd_heading      2. body pose: height, pitch, roll
                             3. leveling PID + heading hold (IMU)
                             4. inverse kinematics in the joint region
                             5. leg pair collision guard
                                   | /servo_cmd
                                   v
                             servo_driver: clamp, rate limit, watchdog
                                   v
                             PCA9685 --> 8 servos
```

The controller core (`GaitControllerCore`) has no ROS dependency; the `gait_controller` node is
a thin wrapper around it. In simulation the `sim` node replaces the IMU and feeds the driver's
`/servo_state` into the MuJoCo hinges. Details are in
[ros2_ws/docs/control.md](ros2_ws/docs/control.md) and [docs/](docs/README.md).

## Repository

| Path | Contents |
|---|---|
| [`ros2_ws/`](ros2_ws) | ROS 2 Humble packages, Pixi workspace, deploy terminal |
| [`ros2_ws/src/robot_dog_control`](ros2_ws/src/robot_dog_control) | kinematics, gait, pose, controller node, behavior API, gamepad, calibration tools |
| [`ros2_ws/src/robot_dog_hw`](ros2_ws/src/robot_dog_hw) | PCA9685 servo driver, IMU supervisor, servo tools |
| [`ros2_ws/src/robot_dog_bringup`](ros2_ws/src/robot_dog_bringup) | launch files and `config/robot_dog.yaml`, the only configuration file |
| [`ros2_ws/src/robot_dog_description`](ros2_ws/src/robot_dog_description) | URDF, meshes, leg collision data, Foxglove layout |
| [`ros2_ws/src/robot_dog_sim`](ros2_ws/src/robot_dog_sim) | MuJoCo model and simulation node |
| [`ros2_ws/src/ext/smbus_compat`](ros2_ws/src/ext/smbus_compat) | `smbus` shim over `smbus2` for the BNO055 driver |
| [`cad/`](cad), [`pcb/`](pcb) | STEP files of the printed parts; servo shield schematic, layout and BOM |
| [`docs/`](docs/README.md) | documentation index |

## Roadmap

- Reinforcement-learning locomotion policy, trained in the MuJoCo simulation and run through the existing controller and safety layers

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

[MuJoCo](https://mujoco.org), [ROS 2](https://docs.ros.org/en/humble/) and
[RoboStack](https://robostack.github.io), [Pixi](https://pixi.sh),
[`bno055`](https://github.com/flynneva/bno055) ROS driver, [smbus2](https://github.com/kplindegaard/smbus2),
and [Foxglove](https://foxglove.dev) for visualization.

## License

[MIT](LICENSE) © [Zerun Wang](https://github.com/ZerunW)

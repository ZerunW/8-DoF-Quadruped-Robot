# Robot Dog Control (ROS 2)

ROS 2 Humble control stack for the 8-DoF quadruped. Each leg is a planar five-bar linkage
driven by two 360° servos. The same code runs on the robot's Raspberry Pi 5 and in a MuJoCo
simulation on a laptop.

![Stand pose](src/robot_dog_description/doc/stand_pose.png)

## Features

- **Kinematics**: exact forward kinematics of the five-bar leg and Gauss-Newton IK, limited
  to a feasible joint region computed from the CAD.
- **Gait**: one generator for all gaits. Gaits are data (phase offsets, duty, period, Bézier
  swing), and the commanded speed selects the gait through a speed schedule.
- **Body pose**: height, pitch and roll as one rigid transform, clamped to validated limits.
- **Balance**: IMU roll/pitch leveling and IMU heading hold while walking.
- **Safety**: startup envelope check of every plannable pose, a per-tick front/rear leg
  collision guard, and a servo driver with a hard clamp, rate limiter and watchdog.
- **Interfaces**: a small Python behavior API (`walk`, `step_in_place`, `set_height`,
  `set_lean`), PS5 DualSense teleoperation, and plain ROS topics.
- **Simulation**: MuJoCo model of the full CAD, driven by the unchanged driver and
  controller.
- **Tooling**: a [Pixi](https://pixi.sh) workspace on `robostack-humble`, so neither the
  laptop nor the Pi needs a system ROS install. A deploy terminal syncs and runs everything
  on the Pi from the laptop.

## Hardware

| Part | |
|---|---|
| Computer | Raspberry Pi 5, Ubuntu 24.04 |
| Servos | 8 × Feetech FT6335 (360°, PWM) |
| Servo driver | PCA9685 on I2C bus 1, address 0x40 |
| IMU | BNO055 on I2C bus 1, address 0x28 |
| Gamepad (optional) | PS5 DualSense over Bluetooth |

The PCA9685 and BNO055 are powered from the servo rail, so they only answer on I2C while
motor power is on. Wiring, servo channels and the IMU mount are in
[docs/hardware.md](docs/hardware.md).

## Packages

```
src/
  robot_dog_hw/           servo driver node (PCA9685), IMU supervisor, servo tools
  robot_dog_control/      kinematics, gait, pose, controller node, behavior API, gamepad, calibration tools
  robot_dog_bringup/      launch files and config/robot_dog.yaml (the only configuration file)
  robot_dog_description/  URDF, meshes, leg collision set, 3D view
  robot_dog_sim/          MuJoCo model and sim node
  ext/smbus_compat/       smbus shim for the BNO055 driver
scripts/
  deploy.py               deploy terminal for the Pi
  pair_gamepad.sh         DualSense pairing
```

Data flow:

```
gamepad / API / ros2 topic pub
        │  /cmd_vel /cmd_step /cmd_pose /cmd_heading
        ▼
gait_controller  ◄── /bno055/imu
        │  /servo_cmd (50 Hz)
        ▼
servo_driver ──► PCA9685 ──► servos
```

## Installation

Install Pixi on the laptop (linux-64) and on the Pi (linux-aarch64):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then, in this folder on the laptop:

```bash
pixi install
pixi run build
pixi run test
```

On the Pi, enable I2C (`dtparam=i2c_arm=on` in the boot config) and add your user to the
`i2c` group.

## Running in simulation

```bash
pixi run launch_sim            # or launch_sim_viewer for the MuJoCo window
```

Connect [Foxglove](https://foxglove.dev) to `ws://localhost:8765` and import
`src/robot_dog_description/foxglove/robot_dog_layout.json`. In a second shell:

```bash
pixi run walk_test -- --speed 0.08 --duration 5
pixi run sim_push -- --fy 3 --duration 0.2
```

See [docs/simulation.md](docs/simulation.md).

## Running on the robot

The laptop holds the workspace. The deploy terminal syncs it to the Pi with rsync and runs
Pixi tasks there over SSH:

```text
pixi run deploy
robotdog> config                    # host, user, deploy directory, SSH key
robotdog> pixi run build            # sync and build on the Pi
```

### First power-up

With the dog on a stand and the feet free:

```text
robotdog> pixi run launch_driver    # servo driver only, holds the stand pose
robotdog> pixi run servo_tune       # adjust servo offsets
robotdog> pixi run leg_test -- --leg lf --sweep-z -15 15    # check each leg's direction
robotdog> pixi run launch_bringup   # full stack: stands up and levels
robotdog> pixi run imu_check        # check the IMU signs
robotdog> pixi run calib_offsets    # level the body and store the servo offsets
```

The calibration tools print the new values, and the deploy terminal writes them into
`config/robot_dog.yaml` on the laptop. The full procedure is in
[docs/bringup.md](docs/bringup.md).

### Walking

```text
robotdog> pixi run launch_bringup
robotdog> pixi run walk_test -- --speed 0.05 --duration 5
```

To stop, run `pixi run stand`. To freeze, call
`ros2 service call /gait_controller/estop std_srvs/srv/Trigger`. To turn the servos off, run
`pixi run relax`.

### Gamepad

Pair the DualSense once with `robotdog> pixi run pair_gamepad`, then:

```text
robotdog> pixi run launch_bringup teleop:=true
```

| Control | Action |
|---|---|
| L1 (hold) | deadman; the dog steps in place and the sticks drive it |
| Left / right stick | left / right side forward or backward (tank drive) |
| R1 / R2 | raise / lower the body |
| Cross | emergency stop |
| Options | stand, re-arm after an emergency stop |

### Foxglove

```text
robotdog> pixi run launch_foxglove
```

Connect to `ws://<pi-host>:8765`. Open the port once with `robotdog> pixi run pi_setup`.

## Behavior API

```python
from robot_dog_control.api import RobotDog

dog = RobotDog()                                        # heading 0 = where it faces now
dog.walk(height=155, heading=0, speed=0.08, duration=5) # mm, deg, m/s, s
dog.walk(speed=-0.05, distance=0.3)                     # back up 0.3 m
dog.step_in_place(duration=4, heading=90)               # turn in place to 90°
dog.set_height(175); dog.set_height(130)                # push-up
dog.set_lean(pitch_deg=10, roll_deg=0)                  # +pitch = nose down
dog.close()
```

The API needs the controller running, on the robot or in simulation. The commanded speed
picks the gait; there is no separate gait command.

## Configuration

Everything is configured in `src/robot_dog_bringup/config/robot_dog.yaml`: servo channels,
offsets and limits, leg geometry, joint region, gait presets, pose limits, controller gains
and gamepad mapping. Most controller parameters can be changed live with
`ros2 param set /gait_controller <name> <value>`, and a change that would make any planned
pose unsafe is rejected.

## Documentation

- [Hardware](docs/hardware.md): I2C devices, servo mapping, IMU mount, gamepad setup
- [Bring-up and calibration](docs/bringup.md): deploy terminal, first power-up, tuning
- [Control](docs/control.md): controller pipeline, conventions, topics, services
- [Safety](docs/safety.md): servo limits and leg collision protection
- [Robot model](docs/model.md): URDF, joint region, collision set
- [Simulation](docs/simulation.md): MuJoCo simulation

## License

[MIT](../LICENSE)

# Bring-up and calibration

## Deploy terminal

`pixi run deploy` opens a terminal (`robotdog>`) that rsyncs the repository to the Pi and runs
commands there:

| Command | |
|---|---|
| `config` | set host, user and deploy directory, install an SSH key |
| `pixi run <task>` | sync, then run the task on the Pi |
| `remote` | SSH shell in the deploy directory |
| `scp pull <dir>` | copy a directory back from the Pi |
| `service kill` | stop everything started from the deploy directory |

The defaults are user `pi`, host `robotdog.local` and `~/robot_dog_ws`. `config` stores your
own values in `scripts/.deploy_config` (git-ignored) and uses a dedicated key,
`~/.ssh/id_ed25519_robotdog`.

Calibration tools print `ROBOT_DOG_CONFIG_UPDATE {...}` lines, and the deploy terminal writes
those values into the laptop's `robot_dog.yaml`.

## First power-up

Put the dog on a stand with the feet free.

1. `pixi run launch_driver`: the driver initializes the PCA9685 and holds the stand pose.
2. `pixi run servo_tune`: adjust servo offsets.
3. `pixi run leg_test -- --leg lf --sweep-z -15 15`: raising `fz` must lift the foot, and
   `--sweep-x -15 15` must move it toward the nose with rising `fx`. Repeat for each leg.
4. `pixi run launch_bringup`: the controller blends to its stand pose and starts leveling.
5. `pixi run imu_check`: nose down gives positive pitch, right side down positive roll,
   turning left positive yaw. Fix signs with `placement_axis_remap` in the `bno055` section.

## Calibration

`pixi run calib_offsets` levels the body with the IMU, lets you align the feet fore/aft
(`--align right=+25`), and folds the result into `servo_offset`. If the IMU board is not
flat in the chassis, level by eye with `--raise` / `--align` and `--no-fold`, then record the
pose with `--level-here`.

## First walk

With `launch_bringup` running, drive the dog with the gamepad:

```text
robotdog> pixi run launch_teleop
```

The behavior API in [control.md](control.md#behavior-api) works the same way, for example
`dog.walk(speed=0.03, duration=5)`.

`pixi run stand` stops the dog, `pixi run relax` turns the servos off, and
`ros2 service call /gait_controller/estop std_srvs/srv/Trigger` freezes it.

## Tuning

| Symptom | Parameter |
|---|---|
| body oscillates in roll/pitch | lower `level_kp`, raise `level_kd` |
| servos twitch while standing | `level_deadband_deg` |
| slow to level | raise `level_ki` |
| leveling fights the gait | lower `level_walk_gain_scale` |
| curves while walking | raise `heading_kp` |
| heading hunts | lower `heading_kp`, raise `heading_kd` |
| feet scuff | raise `step_height_mm` |
| body too low or leaning | `stand_height_mm`, `stand_fx_mm` |

If you change a gait preset, keep servo motion at or below 5.6° per 20 ms tick. The servo
driver's rate limit (400°/s, 8° per tick) is the hard bound.

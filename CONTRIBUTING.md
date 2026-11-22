# Contributing

Bug reports, simulation results and pull requests are welcome.

## Set up

Everything runs from `ros2_ws/` in a [Pixi](https://pixi.sh) environment, so no system ROS 2
install is needed:

```bash
cd ros2_ws
pixi install
pixi run build
pixi run test
```

Try changes in simulation first (`pixi run launch_sim`, then drive it with the behavior API).
The simulation runs the same driver and controller as the robot. See
[ros2_ws/docs/simulation.md](ros2_ws/docs/simulation.md).

## Checks

From the repository root:

```bash
pip install ruff
ruff check .
```

The ruff configuration in [`ruff.toml`](ruff.toml) enables correctness rules only. CI runs
the same check.

## Code style

- Python 3.12, type hints on public functions.
- Keep ROS-free logic in the core classes (`GaitControllerCore`, kinematics, gait). Node
  files should stay thin.
- Parameters belong in `src/robot_dog_bringup/config/robot_dog.yaml`.
- Comments explain why (units, hardware quirks, frame conventions), not what the line does.
- Leg order is `lf, rf, lb, rb` and the body frame is REP-103. See
  [ros2_ws/docs/control.md](ros2_ws/docs/control.md#conventions).

## Changes that touch the hardware

Anything that changes servo limits, the joint region, pose limits or the leg collision set
needs a note in the pull request on how it was checked (startup envelope check, simulation,
robot). Test on a stand with the feet free before putting the dog on the ground.

## Pull requests

- One change per pull request, with a short description of what and why.
- Run `pixi run test` when you touch the servo driver or the leg collision guard. The tests
  in `robot_dog_hw/test` and `robot_dog_control/test` cover those two safety layers.
- Update the docs if topics, parameters or procedures change.

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).

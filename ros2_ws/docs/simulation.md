# Simulation

```bash
pixi run launch_sim           # headless, Foxglove bridge on ws://localhost:8765
pixi run launch_sim_viewer    # with the MuJoCo viewer window
ros2 launch robot_dog_sim sim.launch.py ground_tilt_pitch_deg:=3 payload_mass_kg:=0.3
```

The real `servo_driver` (with a fake PCA9685) and `gait_controller` run unchanged. The `sim`
node replaces the IMU, and the driver's `/servo_state` drives the motor hinges.

The model `robot_dog_sim/mjcf/robot_dog.xml` is generated from the URDF by
`pixi run build_mjcf`. It has a free base, 24 hinges, 8 equality constraints closing the leg
loops, and sphere feet. Servos are position actuators with a ±2 N·m limit and a 400°/s
slew, the same as the driver's rate limit. Physical parameters can be
changed live, for example `ros2 param set /sim servo_kp 40.0`.

| Topic / service | |
|---|---|
| `/bno055/imu` | simulated IMU |
| `/sim/ground_truth` | body pose and twist |
| `/sim/foot_forces` | normal force per foot (N) |
| `/joint_states`, TF `odom -> base_link` | ground truth |
| `/sim/apply_wrench` | Wrench on the body |
| `/sim/push`, `/sim/reset` | Trigger |

Live parameters of `/sim` include `ground_tilt_roll_deg`, `ground_tilt_pitch_deg`,
`payload_mass_kg`, `servo_kp`, `servo_kv`, `servo_torque_nm` and `imu_noise_deg`.

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05}}"
pixi run sim_push -- --fy 3 --duration 0.2
pixi run sim_reset
```

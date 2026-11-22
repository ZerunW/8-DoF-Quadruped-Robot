# Security

The deploy terminal (`ros2_ws/scripts/deploy.py`) uses SSH and rsync to a Raspberry Pi, and
`pixi run pi_setup` runs `sudo ufw allow 8765/tcp` there for the Foxglove bridge. ROS 2
discovery is limited to localhost by default (`ROS_LOCALHOST_ONLY=1`). The Foxglove
WebSocket has no authentication, so only expose it on a trusted network.

Report vulnerabilities privately through GitHub's "Report a vulnerability" button on the
Security tab of this repository, rather than in a public issue.

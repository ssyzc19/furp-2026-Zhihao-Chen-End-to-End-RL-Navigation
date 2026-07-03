"""
Discrete PointNav action -> TurtleBot3 /cmd_vel Twist converter.

Habitat PointNav has 4 discrete actions:
    0 = STOP
    1 = MOVE_FORWARD (0.25 m)
    2 = TURN_LEFT    (10 deg)
    3 = TURN_RIGHT   (10 deg)

On the real robot we approximate each discrete action by publishing a
constant Twist for a fixed duration (open-loop).  The durations are derived
from the commanded speed so that speed * duration == the Habitat step size.

IMPORTANT: measure the REAL forward/turn speed on your robot and update
LINEAR_SPEED / ANGULAR_SPEED below — wheel slip and controller lag mean the
commanded speed is rarely the actual speed.  See README for the calibration
procedure.
"""

import math

# Habitat step sizes (from pointnav config)
FORWARD_STEP_M = 0.25          # forward_step_size
TURN_STEP_RAD = math.radians(10.0)   # turn_angle = 10 deg

# Commanded velocities (m/s, rad/s) — CALIBRATE THESE ON THE REAL ROBOT
LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.5

# Discrete action ids (must match Habitat action space ordering)
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


def action_to_twist_cmd(action_id):
    """
    Map a discrete action id to (linear_x, angular_z, duration_sec).

    Returns (0.0, 0.0, 0.0) for STOP — the caller should treat that as
    "episode finished, stop the robot".
    """
    if action_id == MOVE_FORWARD:
        duration = FORWARD_STEP_M / LINEAR_SPEED
        return LINEAR_SPEED, 0.0, duration
    elif action_id == TURN_LEFT:
        duration = TURN_STEP_RAD / ANGULAR_SPEED
        return 0.0, ANGULAR_SPEED, duration
    elif action_id == TURN_RIGHT:
        duration = TURN_STEP_RAD / ANGULAR_SPEED
        return 0.0, -ANGULAR_SPEED, duration
    else:  # STOP or unknown
        return 0.0, 0.0, 0.0


ACTION_NAMES = {
    STOP: "STOP",
    MOVE_FORWARD: "MOVE_FORWARD",
    TURN_LEFT: "TURN_LEFT",
    TURN_RIGHT: "TURN_RIGHT",
}

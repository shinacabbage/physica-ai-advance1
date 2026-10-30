"""Planar diff-drive features for the course task.

ACT needs a camera image or observation.environment_state. This task has no
camera, so the same planar odometry is stored in both observation.state and
observation.environment_state. Training normalizes both.
"""

import math

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (3,),
        "names": [
            "position_x",
            "position_y",
            "yaw",
        ],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (3,),
        "names": [
            "position_x",
            "position_y",
            "yaw",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": [
            "linear_velocity_x",
            "angular_velocity_z",
        ],
    },
}

robot_type = "diff_drive"
repo_id = "demoRobot_teleop"
root = "/home/ubuntu/dataset/lerobot_dataset/demoRobot_teleop"
task_name = "drive_to_goal"


def yaw_from_quaternion(qx, qy, qz, qw):
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


def state_from_odometry(msg):
    """x, y, yaw only.

    Forward speed and yaw rate are omitted. In these recordings they match
    cmd_vel, so a policy that sees them just repeats the current velocity and
    never starts moving from a stop.
    """
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    yaw = yaw_from_quaternion(
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    return [position.x, position.y, yaw]

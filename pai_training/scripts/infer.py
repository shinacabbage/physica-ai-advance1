#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from lerobot.common.policies.act.modeling_act import ACTPolicy


def load_feature_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "car_features.py"
    spec = importlib.util.spec_from_file_location("car_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__("demo_robot_act_infer")
        self.state_from_odometry = load_feature_config().state_from_odometry

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.subscription = self.create_subscription(
            Odometry,
            "/model/demoRobot/odometry",
            self.odom_callback,
            10,
        )

        self.device = torch.device("cpu")
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

        self.latest_state = None
        self.step_count = 0
        # Same rate as the converted dataset (20 Hz).
        self.timer = self.create_timer(0.05, self.publish_action)

    def odom_callback(self, msg):
        self.latest_state = np.asarray(
            self.state_from_odometry(msg), dtype=np.float32
        )

    def publish_action(self):
        if self.latest_state is None:
            return

        state = torch.from_numpy(self.latest_state.copy()).to(self.device).unsqueeze(0)
        obs = {
            "observation.state": state,
            "observation.environment_state": state.clone(),
        }

        try:
            with torch.no_grad():
                # select_action normalizes, predicts, and unnormalizes.
                # With n_action_steps=1 this is the current command, action chunk index 0.
                action = self.policy.select_action(obs).cpu().numpy().reshape(-1)

            if action.shape != (2,):
                self.get_logger().error(
                    f"Expected 2 action values, got shape {tuple(action.shape)}"
                )
                return

            cmd = Twist()
            cmd.linear.x = float(action[0])
            cmd.angular.z = float(action[1])
            self.cmd_pub.publish(cmd)

            self.step_count += 1
            if self.step_count % 20 == 0:
                self.get_logger().info(
                    f"state={np.array2string(self.latest_state, precision=3)} "
                    f"linear={action[0]:.3f} angular={action[1]:.3f}"
                )
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            self.stop_robot()
            raise

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="ACT inference for demoRobot")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    rclpy.init()
    node = InferenceNode(args.model_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

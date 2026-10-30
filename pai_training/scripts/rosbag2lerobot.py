#!/usr/bin/env python3

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


ODOM_TOPIC = "/model/demoRobot/odometry"
CMD_TOPIC = "/cmd_vel"


def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("robot_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=20):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq

        self.episode_dirs = sorted({
            p.parent for p in self.bag_path.rglob("*.db3")
        })
        if not self.episode_dirs:
            raise FileNotFoundError(
                f"No rosbag2 episode directories found in {self.bag_path}"
            )

        config = load_config_py(config_path)
        self.features = config.features
        self.state_from_odometry = config.state_from_odometry
        self.robot_type = getattr(config, "robot_type", "diff_drive")
        self.repo_id = getattr(config, "repo_id", "demoRobot_teleop")
        self.task_name = getattr(config, "task_name", "drive_to_goal")

        print(f"Found {len(self.episode_dirs)} episode(s)")
        print(f"Output: {self.output_dir}")
        print(f"Target frequency: {self.target_freq} Hz")

    def _get_messages(self, episode_dir):
        messages = {ODOM_TOPIC: [], CMD_TOPIC: []}

        with Reader(episode_dir) as reader:
            connections = [
                c for c in reader.connections
                if c.topic in messages
            ]

            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((timestamp, msg))

        for topic in messages:
            messages[topic].sort(key=lambda item: item[0])

        return messages

    @staticmethod
    def _sample_and_hold(times, values, target_times):
        """Use the most recent sample at or before each target time."""
        indices = np.searchsorted(times, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(times) - 1)
        return values[indices]

    def _read_episode(self, episode_dir):
        messages = self._get_messages(episode_dir)
        odom = messages[ODOM_TOPIC]
        cmds = messages[CMD_TOPIC]

        if not odom:
            raise ValueError(f"No odometry messages in {episode_dir}")
        if not cmds:
            raise ValueError(f"No cmd_vel messages in {episode_dir}")

        odom_times = np.array([t for t, _ in odom], dtype=np.int64)
        cmd_times = np.array([t for t, _ in cmds], dtype=np.int64)

        state_values = [self.state_from_odometry(msg) for _, msg in odom]

        action_values = []
        for _, msg in cmds:
            action_values.append([
                msg.linear.x,
                msg.angular.z,
            ])

        state_values = np.asarray(state_values, dtype=np.float32)
        action_values = np.asarray(action_values, dtype=np.float32)

        # Use only the time interval covered by both topics.
        start = max(odom_times[0], cmd_times[0])
        end = min(odom_times[-1], cmd_times[-1])

        if end <= start:
            raise ValueError("Odometry and cmd_vel have no overlapping time range")

        period_ns = int(1_000_000_000 / self.target_freq)
        target_times = np.arange(start, end, period_ns, dtype=np.int64)

        if len(target_times) == 0:
            raise ValueError("Episode is too short to produce any frames")

        states = self._sample_and_hold(
            odom_times, state_values, target_times
        )
        actions = self._sample_and_hold(
            cmd_times, action_values, target_times
        )
        states, actions = self._trim_stopped_ends(states, actions)

        return states, actions

    @staticmethod
    def _trim_stopped_ends(states, actions):
        """Drop the wait at the start and the stop at the end of a recording."""
        moving = np.any(np.abs(actions) > 1e-6, axis=1)
        if not np.any(moving):
            raise ValueError("Episode contains no non-zero cmd_vel")
        start = int(np.argmax(moving))
        end = len(moving) - int(np.argmax(moving[::-1]))
        if start > 0 or end < len(actions):
            print(f"Trimmed {start} leading and {len(actions) - end} trailing stopped frames")
        return states[start:end], actions[start:end]

    def convert(self):
        # Refuse to overwrite an existing dataset automatically.
        if self.output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {self.output_dir}\n"
                "Choose a new output_dir or move the old directory first."
            )

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.target_freq,
            root=self.output_dir,
            robot_type=self.robot_type,
            features=self.features,
            use_videos=False,
        )

        dataset.meta.add_task(self.task_name)

        successful = 0

        for episode_index, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing episode {episode_index}: {episode_dir}")

            try:
                states, actions = self._read_episode(episode_dir)

                if len(states) != len(actions):
                    raise ValueError("State/action frame counts do not match")

                for state, action in zip(states, actions):
                    state_tensor = torch.from_numpy(state.copy())

                    dataset.add_frame({
                        "observation.state": state_tensor,
                        "observation.environment_state": state_tensor.clone(),
                        "action": torch.from_numpy(action.copy()),
                        "task": self.task_name,
                    })

                dataset.save_episode()
                successful += 1
                print(f"Saved {len(states)} frames")

            except Exception as exc:
                print(f"Episode conversion failed: {exc}")
                raise

        if successful == 0:
            raise RuntimeError("No episodes were converted")

        print(f"\nDataset saved to: {self.output_dir}")
        print(f"Successfully converted episodes: {successful}")
        return str(self.output_dir)


def main():
    with open("configs/conversion_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    converter = RosbagToLeRobot(
        bag_dir=cfg["bag_dir"],
        output_dir=cfg["output_dir"],
        config_path=cfg["config"],
        target_freq=cfg.get("target_freq", 20),
    )
    converter.convert()


if __name__ == "__main__":
    main()


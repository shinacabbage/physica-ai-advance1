# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""
Train ACT (Action Chunking with Transformer) policy on the collected dataset.

This script trains an ACT policy using configuration parameters specified in:
    - configs/train_config.yaml: Training hyperparameters like batch size, learning rate, etc.
    - configs/car_features.py: Dataset and feature configuration

Usage:
    pixi run train

"""


from pathlib import Path
import importlib.util
import wandb
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, NormalizationMode

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("car_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# =====================
# Load parameters from YAML
# =====================
train_cfg = load_train_config("configs/train_config.yaml")
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(train_cfg["output_directory"])
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
CHUNK_SIZE = train_cfg["chunk_size"]
N_ACTION_STEPS = train_cfg["n_action_steps"]
LEARNING_RATE = train_cfg["learning_rate"]
DELTA_TIMESTAMPS = train_cfg["delta_timestamps"]
# =====================

def _latest_observation(value):
    """Drop the time axis added by delta_timestamps. ACT expects (B, D)."""
    if value.ndim == 3:
        return value[:, -1, :]
    return value


def _check_action_timestamps(delta_timestamps, fps, chunk_size):
    action_dt = delta_timestamps["action"]
    if len(action_dt) != chunk_size:
        raise ValueError(
            f"len(delta_timestamps['action'])={len(action_dt)} but chunk_size={chunk_size}"
        )
    if abs(action_dt[0]) > 1e-6:
        raise ValueError(
            "action delta_timestamps must start at 0. "
            "A leading -0.1 is the 10 Hz example and trains the policy to emit the previous command."
        )
    frame_dt = 1.0 / fps
    for index, timestamp in enumerate(action_dt):
        if abs(timestamp - index * frame_dt) > 1e-4:
            raise ValueError(
                f"action timestamp {timestamp} is not frame {index} at {fps} Hz "
                f"(expected {index * frame_dt})."
            )


def main():
    # Create a directory to store the training checkpoint.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = load_config_py(CONFIG_PATH)
    features = config.features

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    for key in ("observation.state", "observation.environment_state", "action"):
        expected = tuple(features[key]["shape"])
        actual = tuple(dataset_metadata.features[key]["shape"])
        if actual != expected:
            raise ValueError(
                f"{key} shape in the dataset is {actual}, but {CONFIG_PATH} expects {expected}. "
                "Move the old dataset aside and run `pixi run convert` again."
            )
    _check_action_timestamps(DELTA_TIMESTAMPS, dataset_metadata.fps, CHUNK_SIZE)

    wandb.init(project="pai_demo_robot", config={
        "repo_id": REPO_ID,
        "dataset_root": str(DATASET_ROOT),
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
    })

    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        # ENV is otherwise left unnormalized, so the copied odometry would keep raw units.
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)

    # Instantiate the dataset
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, delta_timestamps=DELTA_TIMESTAMPS)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch["observation.state"] = _latest_observation(batch["observation.state"])
            batch["observation.environment_state"] = _latest_observation(
                batch["observation.environment_state"]
            )
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    policy.save_pretrained(OUTPUT_DIRECTORY)
    print(f"policy saved to {OUTPUT_DIRECTORY}")
    wandb.finish()

if __name__ == "__main__":
    main()

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

SENSORS = ("gps", "imu", "lidar", "camera")

def compute_normalization(features, train_end):
    result = {}
    for sensor in SENSORS:
        x = np.asarray(features[sensor][:train_end], dtype=np.float32)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        result[sensor] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return result

def apply_normalization(features, normalization):
    return {
        sensor: ((np.asarray(features[sensor], dtype=np.float32) - normalization[sensor]["mean"])
                 / normalization[sensor]["std"]).astype(np.float32)
        for sensor in SENSORS
    }

class FactorReliabilityV2Dataset(Dataset):
    def __init__(self, features, current_labels, future_labels, frame_ids, sequence_length=64):
        self.features = {s: torch.from_numpy(np.asarray(features[s], dtype=np.float32)) for s in SENSORS}
        self.current_labels = {s: torch.from_numpy(np.asarray(current_labels[s], dtype=np.float32)) for s in SENSORS}
        self.future_labels = {s: torch.from_numpy(np.asarray(future_labels[s], dtype=np.float32)) for s in SENSORS}
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.sequence_length = int(sequence_length)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame_id = int(self.frame_ids[index])
        start = frame_id - self.sequence_length + 1
        if start < 0:
            raise IndexError("frame_id does not have enough history")

        item = {"frame_id": frame_id}
        for sensor in SENSORS:
            item[f"{sensor}_feature"] = self.features[sensor][start:frame_id + 1]
            item[f"{sensor}_current_label"] = self.current_labels[sensor][frame_id]
            item[f"{sensor}_future_label"] = self.future_labels[sensor][frame_id]
        return item

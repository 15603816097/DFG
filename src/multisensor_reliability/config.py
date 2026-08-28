from dataclasses import dataclass

@dataclass(frozen=True)
class MultiSensorReliabilityConfig:
    horizon: int = 3
    sequence_length: int = 64
    dt: float = 0.1
    random_seed: int = 20260826
    reliability_gamma: float = 1.5
    gps_window: int = 10
    imu_window: int = 20

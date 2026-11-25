from pathlib import Path
import numpy as np
import pytest

from experiments.isj_v8.prepare_v9_sequences import frame_index, pair_disabled, verify_arrays


def test_original_frame_ids_after_missing_scans():
    missing=(177,178,179,180)
    paths=[Path(f"{k:010d}.bin") for k in range(447) if k not in missing]
    mapping=frame_index(paths,447,missing)
    assert len(mapping)==443
    assert mapping[176]==176 and mapping[181]==177 and mapping[446]==442
    assert all(k not in mapping for k in missing)
    assert pair_disabled(447,missing)==[176,177,178,179,180]
    assert 446-len(pair_disabled(447,missing))==441
    assert pair_disabled(474,())==[]


def test_reject_unexpected_missing_frame():
    paths=[Path(f"{k:010d}.bin") for k in (0,1,3,4)]
    with pytest.raises(ValueError):
        frame_index(paths,5,())
    with pytest.raises(ValueError):
        frame_index(paths,5,(3,))


def test_validate_pair_masks_without_reference(tmp_path):
    n=8
    np.savez(tmp_path/"degraded_sensor_data.npz",
             degraded_gps_local=np.zeros((n,3)),degraded_imu_gyro=np.zeros((n,3)),
             timestamps_seconds=np.arange(n)*.1)
    mats=np.repeat(np.eye(4)[None],n-1,axis=0)
    for sensor in ("lidar","camera"):
        valid=np.ones(n-1,dtype=bool)
        if sensor=="lidar":
            valid[2:5]=False
        np.savez(tmp_path/f"degraded_{sensor}_factor_data.npz",
                 coordinate_transforms=mats,between_measurements=mats,
                 valid=valid,quality=valid.astype(float))
    # No reference archive exists at all; validation needs no reference.
    assert verify_arrays(tmp_path,n,[2,3,4])=={"lidar_valid":4,"camera_valid":7}
    with pytest.raises(ValueError,match="disabled factor"):
        verify_arrays(tmp_path,n,[1,2,3,4])


def test_legacy_0016_audit_compatibility():
    from experiments.isj_v8.prepare_v9_sequences import reconcile_historical_audit
    from experiments.isj_v7.cross_sequence_v7 import validate_frozen_time_alignment
    ref=np.arange(279,dtype=float)*.1
    streams={k:ref.copy() for k in ("velodyne","left-camera","right-camera")}
    audit=validate_frozen_time_alignment("0016",ref,streams)
    old={"sequence":"0016"}
    assert reconcile_historical_audit("0016",old,audit)=="recomputed_from_raw_timestamps_legacy_0016"
    assert "timestamp_alignment_audit" not in old
    with pytest.raises(ValueError,match="unexpected missing"):
        reconcile_historical_audit("0020",{},audit)
    with pytest.raises(ValueError,match="differs"):
        reconcile_historical_audit("0016",{"timestamp_alignment_audit":{}},audit)


def test_0020_original_exception_preserved():
    from experiments.isj_v8.prepare_v9_sequences import reconcile_historical_audit
    from experiments.isj_v7.cross_sequence_v7 import validate_frozen_time_alignment
    ref=np.arange(1104,dtype=float)*.1
    stream=ref.copy()
    stream[:3]-=.12
    streams={k:stream.copy() for k in ("velodyne","left-camera","right-camera")}
    audit=validate_frozen_time_alignment("0020",ref,streams)
    assert audit["startup_exception_applied"]
    assert reconcile_historical_audit("0020",{"timestamp_alignment_audit":audit},audit)=="archived_audit_verified_against_raw_timestamps"
    with pytest.raises(ValueError,match="unexpected missing"):
        reconcile_historical_audit("0016",{},audit)

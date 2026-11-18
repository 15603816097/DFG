from __future__ import annotations

"""Development E1: GPS pre-update 3D innovation from past states + odometry.

No evaluation truth, clean GPS, oracle, full-trajectory FGO estimates,
or future measurements can enter this function. At frame t:
  (1) propagate last *post-update* pose with the LiDAR t-1->t relative
      pose (Camera only if LiDAR unusable);
  (2) save GPS[t]-propagated_position as pre-update innovation;
  (3) ONLY AFTER saving innovation, update current position using GPS[t].

The initial heading is fitted on the first 'warmup' noisy GPS positions
and past LiDAR/Camera motion, entirely before the evaluated frames.
Consequently this is a causal diagnostic conditioned on a historical
GPS heading bootstrap, NOT a GPS-independent prior. Long GPS biases
may be absorbed and become undetectable; LiDAR drift or wrong heading
may create false GPS alarms. Geometry agreement is separately reported.

The scalar propagated variance is only a rough motion-uncertainty proxy,
NOT a properly propagated pose covariance. Do NOT describe squared
normalized errors as statistically calibrated NIS/chi-square tests.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CausalInnovationSettings:
    warmup: int = 63
    # Defaults preserve the original E1 behavior and its historical results.
    # Alternatives are source-specific diagnostic policies, not an ISJ method.
    motion_policy: str = "lidar_first"
    lidar_quality_min: float = 0.1
    camera_quality_min: float = 0.1
    gps_sigma_m: float = 5.0
    lidar_step_sigma_m: float = 0.15
    camera_step_sigma_m: float = 0.45
    rotation_drift_sigma_m: float = 0.3
    maximum_gps_update_m: float = 5.0
    pair_disagreement_translation_m: float = 1.0
    pair_disagreement_rotation_rad: float = 0.1745329252

    def __post_init__(self):
        if self.motion_policy not in (
            "lidar_first", "camera_first",
            "camera_rotation_lidar_translation",
        ):
            raise ValueError("Unknown motion_policy")
        if not isinstance(self.warmup,int) or self.warmup<4:
            raise ValueError("warmup must be integer >=4")
        for name in (
            "gps_sigma_m", "lidar_step_sigma_m","camera_step_sigma_m",
            "rotation_drift_sigma_m","maximum_gps_update_m",
            "pair_disagreement_translation_m",
            "pair_disagreement_rotation_rad",
        ):
            v=float(getattr(self,name))
            if not np.isfinite(v) or v<=0:
                raise ValueError(f"{name} must be positive finite")
        for name in ("lidar_quality_min","camera_quality_min"):
            v=float(getattr(self,name))
            if not np.isfinite(v) or not 0<=v<=1:
                raise ValueError(f"{name} must lie in [0,1]")


def _yaw_rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.asarray([[c,-s,0],[s,c,0],[0,0,1]],float)


def _rotation_angle(delta):
    return float(np.arccos(np.clip((np.trace(delta)-1.0)/2.0,-1.0,1.0)))


def _validate(kw):
    gps=np.asarray(kw["gps_local"],float)
    if gps.ndim!=2 or gps.shape[1]!=3:
        raise ValueError("gps_local must have shape (n,3)")
    n=len(gps)
    if n<6:
        raise ValueError("Not enough frames")
    streams={}
    for sensor in ("lidar","camera"):
        b=np.asarray(kw[f"{sensor}_between"],float)
        v=np.asarray(kw[f"{sensor}_valid"])
        q=np.asarray(kw[f"{sensor}_quality"],float)
        if b.shape!=(n-1,4,4) or v.shape!=(n-1,) or q.shape!=(n-1,):
            raise ValueError(f"Invalid {sensor} relative factors")
        if v.dtype.kind!="b":
            raise ValueError(f"{sensor} validity mask must be boolean")
        if not np.isfinite(b).all() or not np.isfinite(q).all():
            raise ValueError(f"Nonfinite {sensor} odometry inputs")
        if np.any((q<0)|(q>1)):
            raise ValueError(f"{sensor} quality out of [0,1]")
        if np.max(np.abs(b[:,3,:]-np.array([0,0,0,1])))>1e-6:
            raise ValueError(f"{sensor} homogeneous pose row invalid")
        streams[sensor]=(b,v,q)
    if not np.isfinite(gps).all():
        raise ValueError("Nonfinite GPS observations")
    return gps,streams


def causal_position_innovation(
    *,gps_local,lidar_between,lidar_valid,lidar_quality,
    camera_between,camera_valid,camera_quality,
    settings:CausalInnovationSettings=CausalInnovationSettings()
):
    kw=locals()
    gps,streams=_validate(kw)
    n=len(gps)
    if n<=settings.warmup:
        raise ValueError("Sequence shorter than warmup")
    lb,lv,lq=streams["lidar"]
    cb,cv,cq=streams["camera"]
    lgood=lv & (lq>=settings.lidar_quality_min)
    cgood=cv & (cq>=settings.camera_quality_min)

    def choose_step(i):
        """Use current/previous physical relative factors; never future GPS.

        Motion code: 0 missing, 1 LiDAR, 2 Camera, 3 LiDAR
        translation + Camera rotation (if both quality-valid).
        """
        has_l=bool(lgood[i])
        has_c=bool(cgood[i])
        if settings.motion_policy=="lidar_first":
            return (lb[i],1) if has_l else ((cb[i],2) if has_c else (None,0))
        if settings.motion_policy=="camera_first":
            return (cb[i],2) if has_c else ((lb[i],1) if has_l else (None,0))
        if has_l and has_c:
            pose=np.asarray(lb[i],float).copy()
            pose[:3,:3]=cb[i,:3,:3]
            return pose,3
        return (cb[i],2) if has_c else ((lb[i],1) if has_l else (None,0))

    # Build a historical body-frame relative trajectory from the first
    # warmup frames; this ONLY uses geometry measured before frame warmup.
    local=np.zeros((settings.warmup,3),float)
    orientation=np.eye(3)
    supported=0
    for t in range(1,settings.warmup):
        i=t-1
        step,_=choose_step(i)
        if step is None:
            local[t]=local[t-1]
            continue
        supported+=1
        local[t]=local[t-1]+orientation@step[:3,3]
        orientation=orientation@step[:3,:3]
    if supported<max(4,settings.warmup//2):
        raise ValueError("Too little historical geometry to initialize heading")

    # Fit only 2D yaw by two-vector Procrustes over past 0..warmup-1.
    # This is an intentional, disclosed past-GPS alignment; no GT.
    x=local[:,:2]-local[:,:2].mean(axis=0)
    y=gps[:settings.warmup,:2]-gps[:settings.warmup,:2].mean(axis=0)
    cross=float(np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0]))
    dot=float(np.sum(x*y))
    if float(np.sum(x*x))<1e-6 or np.hypot(cross,dot)<1e-8:
        raise ValueError("Historical GPS/odometry heading not identifiable")
    yaw=float(np.arctan2(cross,dot))
    R=_yaw_rotation(yaw)@orientation
    state_p=gps[settings.warmup-1].copy()
    variance=settings.gps_sigma_m**2

    pred=np.full((n,3),np.nan)
    innov=np.full((n,3),np.nan)
    score=np.full(n,np.nan)
    scale=np.full(n,np.nan)
    mask=np.zeros(n,bool)
    geometry_confident=np.zeros(n,bool)
    disagreement_trans=np.full(n,np.nan)
    disagreement_rot=np.full(n,np.nan)
    mode=np.zeros(n,np.int8)  # 0 missing, 1 lidar, 2 camera, 3 hybrid

    for t in range(settings.warmup,n):
        i=t-1
        l_ok=bool(lgood[i])
        c_ok=bool(cgood[i])
        if l_ok and c_ok:
            dt=float(np.linalg.norm(lb[i,:3,3]-cb[i,:3,3]))
            da=_rotation_angle(lb[i,:3,:3].T@cb[i,:3,:3])
            disagreement_trans[t]=dt
            disagreement_rot[t]=da
            geometry_confident[t]=(
                dt<=settings.pair_disagreement_translation_m
                and da<=settings.pair_disagreement_rotation_rad
            )
        step,motion_code=choose_step(i)
        if step is None:
            # No motion-based prior: skip score; accept after-current
            # GPS ONLY to seed possible future predictions.
            state_p=gps[t].copy()
            variance=settings.gps_sigma_m**2
            continue

        mode[t]=motion_code
        prior=state_p+R@step[:3,3]
        R=R@step[:3,:3]
        process_sigma=(
            settings.lidar_step_sigma_m if motion_code in (1,3)
            else settings.camera_step_sigma_m
        )
        variance+=process_sigma**2+settings.rotation_drift_sigma_m**2
        discrepancy=gps[t]-prior
        length=float(np.linalg.norm(discrepancy))
        pred[t]=prior
        innov[t]=discrepancy
        mask[t]=True
        # Proxy only: not a calibrated full 6x6 innovation covariance.
        scale[t]=float(np.sqrt(
            settings.gps_sigma_m**2+variance
        ))
        score[t]=length/scale[t]

        gain=variance/(variance+settings.gps_sigma_m**2)
        clipped=discrepancy*min(
            1.0,settings.maximum_gps_update_m/max(length,1e-12)
        )
        state_p=prior+gain*clipped
        variance=(1-gain)*variance

    return {
        "prior_position":pred,"innovation":innov,
        "innovation_norm_m":np.linalg.norm(innov,axis=1),
        "standardized_score":score,"scale_proxy_m":scale,
        "valid":mask,"geometry_consistent":geometry_confident,
        "pair_translation_difference_m":disagreement_trans,
        "pair_rotation_difference_rad":disagreement_rot,
        "odometry_source":mode,
        "bootstrapped_heading_rad":yaw,
        "warmup":settings.warmup,
        "motion_policy":settings.motion_policy,
    }

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

EARTH_RADIUS = 6378137.0
TYPES = ("sparse", "range_noise", "occlusion", "ghost_outlier")
LEVELS = ("mild", "moderate", "severe")

@dataclass(frozen=True)
class BenchmarkConfig:
    start_frame: int = 300
    block_length: int = 140
    gap_length: int = 40
    random_seed: int = 20260827

def build_schedule(n, cfg=BenchmarkConfig()):
    type_id = np.zeros(n, np.int32)
    level_id = np.zeros(n, np.int32)
    block_id = np.full(n, -1, np.int32)
    cursor, block = cfg.start_frame, 0
    for tid in range(1, 5):
        for lid in range(1, 4):
            s, e = cursor, min(n, cursor + cfg.block_length)
            if s >= n:
                break
            type_id[s:e], level_id[s:e], block_id[s:e] = tid, lid, block
            cursor, block = e + cfg.gap_length, block + 1
    return dict(type_id=type_id, level_id=level_id, block_id=block_id)

def _rng(frame, cfg, salt):
    return np.random.default_rng(cfg.random_seed + frame*1009 + salt*9176)

def apply_degradation(points, type_id, level_id, frame, cfg=BenchmarkConfig()):
    p = np.asarray(points, np.float64)
    if type_id == 0 or level_id == 0 or len(p) == 0:
        return p.copy()
    xyz = p[:, :3]
    rng = _rng(frame, cfg, type_id*10+level_id)
    if type_id == 1:  # sparse
        ratio = {1:.75, 2:.45, 3:.20}[level_id]
        k = min(len(p), max(200, int(len(p)*ratio)))
        return p[rng.choice(len(p), k, replace=False)].copy()
    if type_id == 2:  # range noise
        sigma = {1:.03, 2:.08, 3:.18}[level_id]
        r = np.linalg.norm(xyz, axis=1)
        d = xyz / np.maximum(r[:,None], 1e-6)
        x = xyz + d*rng.normal(0,sigma,len(p))[:,None] + rng.normal(0,sigma*.15,xyz.shape)
        return np.c_[x, p[:,3:]] if p.shape[1] > 3 else x
    if type_id == 3:  # contiguous azimuth occlusion
        width = np.deg2rad({1:45.,2:90.,3:150.}[level_id])
        a = np.arctan2(xyz[:,1], xyz[:,0])
        center = np.deg2rad(15*np.sin(frame/80.0))
        delta = np.angle(np.exp(1j*(a-center)))
        keep = np.abs(delta) > width/2
        return p[keep].copy() if keep.sum() >= 200 else p.copy()
    if type_id == 4:  # structured ghost / dynamic clutter
        frac = {1:.08,2:.18,3:.32}[level_id]
        shift = {1:[.35,.15,0],2:[.75,.30,.05],3:[1.40,.55,.10]}[level_id]
        k = min(len(p), max(100, int(len(p)*frac)))
        idx = rng.choice(len(p), k, replace=False)
        ghost = p[idx].copy()
        ghost[:,:3] += np.asarray(shift) + np.asarray([0,.15*np.sin(frame/18.),0])
        return np.concatenate([p,ghost],axis=0)
    raise ValueError(type_id)

def _R(roll,pitch,yaw):
    cr,sr=np.cos(roll),np.sin(roll); cp,sp=np.cos(pitch),np.sin(pitch); cy,sy=np.cos(yaw),np.sin(yaw)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx

def load_oxts_world_poses(sequence):
    files=sorted((Path(sequence)/"oxts"/"data").glob("*.txt"))
    if not files: raise FileNotFoundError("OXTS not found")
    rows=[np.fromstring(f.read_text().strip(),sep=" ") for f in files]
    scale=np.cos(rows[0][0]*np.pi/180.)
    poses=[]
    for v in rows:
        lat,lon,alt,roll,pitch,yaw=v[:6]
        tx=scale*lon*np.pi*EARTH_RADIUS/180.
        ty=scale*EARTH_RADIUS*np.log(np.tan((90.+lat)*np.pi/360.))
        T=np.eye(4); T[:3,:3]=_R(roll,pitch,yaw); T[:3,3]=[tx,ty,alt]; poses.append(T)
    poses=np.asarray(poses)
    T0i=np.linalg.inv(poses[0])
    return np.asarray([T0i@T for T in poses])

def gt_between(world_poses):
    return np.asarray([np.linalg.inv(world_poses[i])@world_poses[i+1] for i in range(len(world_poses)-1)])

def lidar_transform_to_body_between(T_lidar_target_source, T_imu_to_velo, T_velo_to_imu):
    coord_body = T_velo_to_imu @ T_lidar_target_source @ T_imu_to_velo
    return np.linalg.inv(coord_body)

def factor_error(est, gt):
    D=np.linalg.inv(gt)@est
    te=float(np.linalg.norm(D[:3,3]))
    c=np.clip((np.trace(D[:3,:3])-1)/2,-1,1)
    re=float(np.degrees(np.arccos(c)))
    return te,re

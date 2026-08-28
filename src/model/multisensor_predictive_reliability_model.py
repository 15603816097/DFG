from __future__ import annotations
import torch
import torch.nn as nn

SENSORS=("gps","imu","lidar","camera")

class SensorEncoder(nn.Module):
    def __init__(self,input_dim,output_dim=32,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,output_dim),nn.GELU())
    def forward(self,x): return self.net(x)

class ReliabilityHead(nn.Module):
    def __init__(self,hidden_dim,dropout=0.1):
        super().__init__()
        self.current=nn.Sequential(nn.Linear(hidden_dim,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1),nn.Sigmoid())
        self.future=nn.Sequential(nn.Linear(hidden_dim,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,h):
        return {"current":self.current(h).squeeze(-1),"future":self.future(h).squeeze(-1)}

class MultiSensorPredictiveReliabilityModel(nn.Module):
    def __init__(self,gps_dim=21,imu_dim=20,lidar_dim=19,camera_dim=16,sensor_embed_dim=32,hidden_dim=128,num_layers=2,dropout=0.1):
        super().__init__()
        dims={"gps":gps_dim,"imu":imu_dim,"lidar":lidar_dim,"camera":camera_dim}
        self.sensor_encoders=nn.ModuleDict({s:SensorEncoder(dims[s],sensor_embed_dim,dropout) for s in SENSORS})
        self.fusion_projection=nn.Sequential(nn.Linear(sensor_embed_dim*4,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout))
        self.backend="gru";self.temporal_encoder=None
        try:
            from src.model.mamba_encoder import MambaEncoder
            for kwargs in (
                {"input_dim":hidden_dim,"hidden_dim":hidden_dim,"num_layers":num_layers},
                {"input_dim":hidden_dim,"d_model":hidden_dim,"num_layers":num_layers},
            ):
                try:
                    self.temporal_encoder=MambaEncoder(**kwargs);self.backend="mamba";break
                except TypeError:
                    self.temporal_encoder=None
        except Exception:
            self.temporal_encoder=None
        if self.temporal_encoder is None:
            self.temporal_encoder=nn.GRU(hidden_dim,hidden_dim,num_layers=num_layers,batch_first=True,dropout=dropout if num_layers>1 else 0.0)
        self.heads=nn.ModuleDict({s:ReliabilityHead(hidden_dim,dropout) for s in SENSORS})
    def _encode(self,x):
        if self.backend=="mamba":
            h=self.temporal_encoder(x)
            if isinstance(h,(tuple,list)): h=h[0]
            if h.ndim==3: h=h[:,-1,:]
            return h
        h,_=self.temporal_encoder(x);return h[:,-1,:]
    def forward(self,gps,imu,lidar,camera):
        xs={"gps":gps,"imu":imu,"lidar":lidar,"camera":camera}
        fused=torch.cat([self.sensor_encoders[s](xs[s]) for s in SENSORS],dim=-1)
        h=self._encode(self.fusion_projection(fused))
        out={"embedding":h}
        for s in SENSORS: out[s]=self.heads[s](h)
        return out

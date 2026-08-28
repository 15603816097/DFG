import numpy as np

def _safe_stats(x):
    x=np.asarray(x,float)
    if x.size==0:return np.zeros(4,np.float32)
    return np.asarray([np.mean(x),np.std(x),np.min(x),np.max(x)],np.float32)

def gps_features(positions,dt=.1,window=10):
    p=np.asarray(positions,float);n=len(p);d=np.zeros_like(p);d[1:]=p[1:]-p[:-1]
    v=d/max(float(dt),1e-6);a=np.zeros_like(v);a[1:]=(v[1:]-v[:-1])/max(float(dt),1e-6);rows=[]
    for i in range(n):
        s=max(0,i-int(window)+1)
        rows.append(np.concatenate([d[i],v[i],a[i],_safe_stats(np.linalg.norm(d[s:i+1],axis=1)),_safe_stats(np.linalg.norm(v[s:i+1],axis=1)),_safe_stats(np.linalg.norm(a[s:i+1],axis=1))]))
    return np.asarray(rows,np.float32)

def imu_features(acceleration,angular_velocity,window=20):
    acc=np.asarray(acceleration,float);gyro=np.asarray(angular_velocity,float);n=min(len(acc),len(gyro));rows=[]
    for i in range(n):
        s=max(0,i-int(window)+1);aw=acc[s:i+1];gw=gyro[s:i+1]
        rows.append(np.concatenate([acc[i],gyro[i],_safe_stats(np.linalg.norm(aw,axis=1)),_safe_stats(np.linalg.norm(gw,axis=1)),np.std(aw,axis=0),np.std(gw,axis=0)]))
    return np.asarray(rows,np.float32)

def lidar_features(points):
    pts=np.asarray(points)
    if pts.ndim!=2 or pts.shape[1]<3:raise ValueError("LiDAR points must be N x 3 or N x 4")
    if len(pts)==0:return np.zeros(19,np.float32)
    xyz=pts[:,:3].astype(float);r=np.linalg.norm(xyz,axis=1);xy=np.linalg.norm(xyz[:,:2],axis=1);z=xyz[:,2]
    intensity=np.zeros(4,np.float32)
    if pts.shape[1]>=4:intensity=_safe_stats(pts[:,3])
    return np.concatenate([np.asarray([len(pts),np.mean(xyz[:,0]>0),np.mean(r<20)],np.float32),_safe_stats(r),_safe_stats(xy),_safe_stats(z),intensity]).astype(np.float32)

def _gray(image):
    img=np.asarray(image).astype(np.float32)
    if img.ndim==2:return img
    if img.ndim==3 and img.shape[2]>=3:return .299*img[:,:,0]+.587*img[:,:,1]+.114*img[:,:,2]
    raise ValueError("Unsupported image shape")

def camera_features(image):
    g=np.clip(_gray(image),0,255)
    if g.size==0:return np.zeros(16,np.float32)
    gx=np.zeros_like(g);gy=np.zeros_like(g);gx[:,1:]=g[:,1:]-g[:,:-1];gy[1:,:]=g[1:,:]-g[:-1,:]
    grad=np.sqrt(gx**2+gy**2)
    lap=-4*g+np.roll(g,1,0)+np.roll(g,-1,0)+np.roll(g,1,1)+np.roll(g,-1,1)
    return np.concatenate([_safe_stats(g),_safe_stats(grad),_safe_stats(lap),np.asarray([np.mean(g<25),np.mean(g>230),np.mean(grad>20),np.std(lap)],np.float32)]).astype(np.float32)

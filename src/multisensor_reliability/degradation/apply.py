import numpy as np

def _rng(seed,frame_id,stream):
    return np.random.default_rng(int(seed)+int(frame_id)*1009+int(stream)*9176)

def corrupt_gps(position,severity,mode,frame_id,seed=20260826):
    p=np.asarray(position,float).copy();s=float(np.clip(severity,0,1))
    if s<=0:return p
    rng=_rng(seed,frame_id,1)
    if int(mode)==2:
        d=rng.normal(size=3);d/=max(np.linalg.norm(d),1e-12);d[2]*=.25;p+=d*(10+18*s)
    else:
        noise=rng.normal(0,.30+4*s,3);noise[2]*=.5
        bias=np.array([6*s*np.sin(frame_id/120),4*s*np.cos(frame_id/155),.8*s*np.sin(frame_id/190)])
        p+=noise+bias
    return p

def corrupt_imu(acceleration,angular_velocity,severity,mode,frame_id,seed=20260826):
    acc=np.asarray(acceleration,float).copy();gyro=np.asarray(angular_velocity,float).copy();s=float(np.clip(severity,0,1))
    if s<=0:return acc,gyro
    rng=_rng(seed,frame_id,2)
    if int(mode)==2:
        acc+=rng.normal(0,2.5*s,3);gyro+=rng.normal(0,.25*s,3)
    else:
        ab=np.array([.45*s*np.sin(frame_id/90),-.35*s*np.cos(frame_id/115),.20*s*np.sin(frame_id/150)])
        gb=np.array([.018*s*np.sin(frame_id/130),-.015*s*np.cos(frame_id/160),.022*s*np.sin(frame_id/100)])
        acc+=ab+rng.normal(0,.12+.75*s,3);gyro+=gb+rng.normal(0,.004+.035*s,3)
    return acc,gyro

def corrupt_lidar(points,severity,mode,frame_id,seed=20260826):
    pts=np.asarray(points).copy()
    if pts.ndim!=2 or pts.shape[1]<3:raise ValueError("LiDAR points must be N x 3 or N x 4")
    s=float(np.clip(severity,0,1))
    if s<=0 or len(pts)==0:return pts
    rng=_rng(seed,frame_id,3);xyz=pts[:,:3]
    if int(mode)==2:
        ang=np.arctan2(xyz[:,1],xyz[:,0]);center=rng.uniform(-np.pi,np.pi);half=.15+.75*s
        wrapped=np.angle(np.exp(1j*(ang-center)));pts=pts[np.abs(wrapped)>half]
        if len(pts):pts[:,:3]+=rng.normal(0,.03+.12*s,(len(pts),3))
    else:
        keep=rng.random(len(pts))<(1-.70*s);pts=pts[keep]
        if len(pts):pts[:,:3]+=rng.normal(0,.02+.18*s,(len(pts),3))
    return pts

def _box_blur(image,radius):
    if radius<=0:return image
    img=np.asarray(image,np.float32);orig2d=(img.ndim==2)
    if orig2d:img=img[:,:,None]
    r=int(radius);k=2*r+1;p=np.pad(img,((r,r),(r,r),(0,0)),mode="reflect");o=np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):o+=p[dy:dy+img.shape[0],dx:dx+img.shape[1],:]
    o/=k*k
    return o[:,:,0] if orig2d else o

def corrupt_camera(image,severity,mode,frame_id,seed=20260826):
    img=np.asarray(image).astype(np.float32).copy();s=float(np.clip(severity,0,1))
    if img.ndim not in (2,3):raise ValueError("Camera image must be HxW or HxWxC")
    if s<=0:return np.clip(img,0,255).astype(np.uint8)
    rng=_rng(seed,frame_id,4);mode=int(mode)
    if mode==1:
        img*=1-.65*s;img=_box_blur(img,int(round(1+4*s)))
    elif mode==2:
        img=img*(1+.85*s)+35*s;img+=rng.normal(0,8+18*s,img.shape)
    else:
        h,w=img.shape[:2];ow=max(1,int(w*(.15+.45*s)));oh=max(1,int(h*(.15+.35*s)))
        x0=int(rng.integers(0,max(1,w-ow+1)));y0=int(rng.integers(0,max(1,h-oh+1)))
        img[y0:y0+oh,x0:x0+ow,...]=0;img+=rng.normal(0,5+10*s,img.shape)
    return np.clip(img,0,255).astype(np.uint8)

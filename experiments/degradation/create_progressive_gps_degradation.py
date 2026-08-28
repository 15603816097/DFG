import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
GT=os.path.join(ROOT,"results","ground_truth","trajectory.txt")
OUT=os.path.join(ROOT,"results","progressive_gps_degradation")
SEED=20260826

def ss(x):
    x=np.clip(x,0,1); return x*x*(3-2*x)

def severity(n):
    s=np.zeros(n); a=np.arange(n)
    for st,rise,hold,end,peak in [(700,1000,1450,1700,.65),(1900,2250,2900,3250,1.0),(3500,3750,4050,4300,.8)]:
        m=(a>=st)&(a<rise); s[m]=np.maximum(s[m],peak*ss((a[m]-st)/max(rise-st,1)))
        m=(a>=rise)&(a<hold); s[m]=np.maximum(s[m],peak)
        m=(a>=hold)&(a<end); s[m]=np.maximum(s[m],peak*(1-ss((a[m]-hold)/max(end-hold,1))))
    return np.clip(s,0,1)

def main():
    gt=np.loadtxt(GT,float); n=len(gt); rng=np.random.default_rng(SEED); sev=severity(n)
    sig=.35+5*sev; noise=rng.normal(size=(n,3))*sig[:,None]; noise[:,2]*=.6
    bias=np.zeros((n,3))
    for i in range(1,n):
        step=rng.normal(0,.015+.10*sev[i],3); step[2]*=.4
        bias[i]=.997*bias[i-1]+sev[i]*step
    t=np.arange(n,float) if False else np.arange(n,dtype=float)
    drift=np.c_[7*sev*np.sin(t/110),5*sev*np.cos(t/145),1.2*sev*np.sin(t/180)]
    gps=gt+noise+bias+drift; err=np.linalg.norm(gps-gt,axis=1)
    os.makedirs(OUT,exist_ok=True)
    np.savetxt(os.path.join(OUT,"gps_corrupted.txt"),gps,fmt="%.8f")
    np.savetxt(os.path.join(OUT,"severity.txt"),sev,fmt="%.8f")
    np.savetxt(os.path.join(OUT,"gps_error.txt"),err,fmt="%.8f")
    print("="*60);print("Progressive GPS degradation");print("Frames:",n)
    print("Error min/max/mean:",err.min(),err.max(),err.mean());print("Saved:",OUT)
if __name__=="__main__": main()

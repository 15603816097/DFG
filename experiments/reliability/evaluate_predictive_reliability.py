import os,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Y=os.path.join(ROOT,"results","predictive_reliability_labels","future_window_reliability_h5.txt")
P=os.path.join(ROOT,"results","predictive_reliability_new","predicted_reliability.txt")
def main():
    y=np.loadtxt(Y);p=np.loadtxt(P);n=min(len(y),len(p));y=y[:n];p=p[:n]
    mae=np.mean(abs(y-p));rmse=np.sqrt(np.mean((y-p)**2))
    corr=np.corrcoef(y,p)[0,1] if np.std(y)>0 and np.std(p)>0 else 0
    tb=y<.3;pb=p<.3;rec=np.sum(tb&pb)/max(np.sum(tb),1);pre=np.sum(tb&pb)/max(np.sum(pb),1)
    starts=np.where(tb & np.r_[True,~tb[:-1]])[0]; leads=[]
    for s in starts:
        a=max(0,s-50);q=np.where(pb[a:s+1])[0]
        if len(q):leads.append((s-(a+q[0]))*.1)
    print("="*60);print("Predictive Reliability Evaluation")
    print(f"MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nCorrelation: {corr:.6f}")
    print(f"BadRecall@0.3: {rec:.6f}\nBadPrecision@0.3: {pre:.6f}")
    print("Detected episodes:",len(leads),"/",len(starts))
    if leads:print("Mean lead time:",np.mean(leads),"s")
if __name__=="__main__":main()

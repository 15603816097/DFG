import os,numpy as np
from fair_fg_common import *
def main():
    gps,a,g,p,_=load_data(pred=True);p=np.clip(p,0,1);run(gps,a,g,sigma(p),os.path.join(ROOT,"results","fair_predictive_only_fg"),"3/5 Predictive Only FG",p,{"predicted_reliability.txt":p})
if __name__=="__main__":main()

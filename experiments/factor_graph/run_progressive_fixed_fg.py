import os
from fair_fg_common import *
def main():
    gps,a,g,_,_=load_data();run(gps,a,g,np.full(len(gps),5.0),os.path.join(ROOT,"results","fair_progressive_fixed_fg"),"1/5 Fixed Covariance FG",huber=False)
if __name__=="__main__":main()

import os
from fair_fg_common import *
def main():
    gps,a,g,_,_=load_data();r,res=reactive(gps);run(gps,a,g,sigma(r),os.path.join(ROOT,"results","fair_current_reliability_fg"),"2/5 Current Reactive Reliability FG",r,{"motion_residual.txt":res})
if __name__=="__main__":main()

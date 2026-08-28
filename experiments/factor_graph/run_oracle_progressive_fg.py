import os
from fair_fg_common import *
def main():
    gps,a,g,_,gt=load_data(gt=True);r,e=oracle(gps,gt);run(gps,a,g,sigma(r),os.path.join(ROOT,"results","fair_oracle_progressive_fg"),"5/5 Oracle Reliability FG",r,{"gps_error.txt":e})
if __name__=="__main__":main()

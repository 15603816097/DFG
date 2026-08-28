import os,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH=os.path.join(ROOT,"results","multisensor_reliability","multisensor_reliability_data.npz")
def main():
    d=np.load(PATH);print("="*84);print("MULTI-SENSOR RELIABILITY DATA INSPECTION")
    for s in ("gps","imu","lidar","camera"):
        f=d[f"{s}_features"];c=d[f"{s}_current_label"];u=d[f"{s}_future_label"];v=d[f"{s}_severity"]
        print(f"\n{s.upper()}");print("features:",f.shape);print("severity:",v.min(),v.max(),v.mean());print("current:",c.min(),c.max(),c.mean());print("future:",u.min(),u.max(),u.mean())
if __name__=="__main__":main()

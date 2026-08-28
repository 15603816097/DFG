import os,sys,subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS=["experiments/multisensor/create_multisensor_degradation_plan.py","experiments/multisensor/build_multisensor_reliability_data.py","experiments/multisensor/inspect_multisensor_reliability_data.py"]
for i,s in enumerate(STEPS,1):
    print(f"\n[{i}/{len(STEPS)}] {s}");subprocess.run([sys.executable,os.path.join(ROOT,s)],cwd=ROOT,check=True)
print("\nMulti-sensor reliability data layer finished.")

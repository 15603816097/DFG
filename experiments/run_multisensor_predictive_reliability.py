import os,sys,subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
steps=[
("Train Shared-Mamba four-sensor reliability model","experiments/multisensor/train_multisensor_predictive_reliability.py"),
("Evaluate target-aligned four-sensor reliability","experiments/multisensor/evaluate_multisensor_predictive_reliability.py"),
]
print("="*92);print("DFG MULTI-SENSOR PREDICTIVE RELIABILITY PIPELINE");print("="*92)
for i,(title,path) in enumerate(steps,1):
    print(f"\n[{i}/{len(steps)}] {title}");subprocess.run([sys.executable,os.path.join(ROOT,path)],cwd=ROOT,check=True)
print("\nPipeline finished.")

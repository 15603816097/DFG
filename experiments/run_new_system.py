import os,sys,subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS=["experiments/degradation/create_progressive_gps_degradation.py","experiments/reliability/create_predictive_reliability_labels.py","experiments/reliability/train_predictive_reliability.py","experiments/reliability/evaluate_predictive_reliability.py","experiments/factor_graph/run_predictive_reliability_fg.py"]
for i,s in enumerate(STEPS,1):
    print(f"\n[{i}/{len(STEPS)}] {s}");subprocess.run([sys.executable,os.path.join(ROOT,s)],cwd=ROOT,check=True)
print("\nNew system finished. Run: python experiments/evaluate_trajectory.py")

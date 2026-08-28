import os,sys,subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS=[
"experiments/factor_graph/run_progressive_fixed_fg.py",
"experiments/factor_graph/run_current_reliability_fg.py",
"experiments/factor_graph/run_predictive_only_fg.py",
"experiments/factor_graph/run_predictive_feedback_fg.py",
"experiments/factor_graph/run_oracle_progressive_fg.py",
"experiments/evaluation/evaluate_fair_ablation.py"]
for i,s in enumerate(STEPS,1):
    print("\n[%d/%d] %s"%(i,len(STEPS),s));subprocess.run([sys.executable,os.path.join(ROOT,s)],cwd=ROOT,check=True)
print("\nAll fair ablation experiments finished.")

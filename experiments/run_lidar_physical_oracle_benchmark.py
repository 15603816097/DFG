from __future__ import annotations
import os,subprocess,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS=[
"experiments/lidar_benchmark/build_lidar_physical_benchmark.py",
"experiments/lidar_benchmark/inspect_lidar_physical_benchmark.py",
"experiments/factor_graph/run_lidar_physical_oracle_ablation.py",
"experiments/evaluation/evaluate_lidar_physical_oracle.py"]
def main():
    print("="*132); print("DFG LIDAR PHYSICAL DEGRADATION + GT ORACLE BENCHMARK"); print("="*132)
    for i,s in enumerate(STEPS,1):
        print("\n"+"#"*132); print(f"[{i}/{len(STEPS)}] {s}"); print("#"*132)
        subprocess.run([sys.executable,os.path.join(ROOT,s)],cwd=ROOT,check=True)
if __name__=="__main__": main()

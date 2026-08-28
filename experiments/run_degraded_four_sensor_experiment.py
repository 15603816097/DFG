import os,sys,subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
steps=[
("Build degraded measurements","experiments/degradation/build_degraded_four_sensor_measurements.py"),
("Degraded fixed FG","experiments/factor_graph/run_degraded_four_sensor_fixed_fg.py"),
("Degraded predictive FG","experiments/factor_graph/run_degraded_four_sensor_predictive_fg.py"),
("Evaluation","experiments/evaluation/evaluate_degraded_four_sensor_fg.py"),
]
print("="*108);print("DFG DEGRADED FOUR-SENSOR END-TO-END EXPERIMENT")
for i,(title,p) in enumerate(steps,1):
    print(f"\n[{i}/{len(steps)}] {title}");subprocess.run([sys.executable,os.path.join(ROOT,p)],cwd=ROOT,check=True)
print("\nEnd-to-end degraded experiment finished.")

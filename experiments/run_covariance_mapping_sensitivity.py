import os,sys,subprocess

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMAS=[1,2,3,4,5]
SIGMA_MAX=[10,15,20,25,30]
BEST_GAMMA=os.path.join(ROOT,"results","covariance_mapping_sensitivity","best_gamma.txt")

def run_fg(g,s,stage):
    subprocess.run([sys.executable,os.path.join(ROOT,"experiments","factor_graph","run_covariance_mapping_fg.py"),
                    "--gamma",str(g),"--sigma-max",str(s),"--stage",stage],cwd=ROOT,check=True)

def run_eval(stage):
    subprocess.run([sys.executable,os.path.join(ROOT,"experiments","evaluation","evaluate_covariance_stage.py"),
                    "--stage",stage],cwd=ROOT,check=True)

print("="*84);print("DFG COVARIANCE MAPPING SENSITIVITY")
print("\nSTAGE 1: gamma sweep | sigma_max=20")
for i,g in enumerate(GAMMAS,1):
    print(f"\n[{i}/{len(GAMMAS)}] gamma={g}")
    run_fg(g,20,"gamma")
run_eval("gamma")

if not os.path.exists(BEST_GAMMA): raise FileNotFoundError(BEST_GAMMA)
best_gamma=float(open(BEST_GAMMA).read().strip())
print("\nSTAGE 2: sigma_max sweep | best gamma =",best_gamma)
for i,s in enumerate(SIGMA_MAX,1):
    print(f"\n[{i}/{len(SIGMA_MAX)}] sigma_max={s}")
    run_fg(best_gamma,s,"sigma_max")
run_eval("sigma_max")
print("\nFinished. See results/best_covariance_mapping.txt")

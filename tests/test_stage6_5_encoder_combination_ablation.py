from pathlib import Path
import ast, importlib.util, numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"experiments/factor_graph/run_stage6_5_encoder_combination_ablation.py"
def mod():
    s=importlib.util.spec_from_file_location("s65",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_syntax(): ast.parse(P.read_text())
def test_cube():
    m=mod(); c=m.build_cases(); assert len(c)==8
    assert len({x[1:] for x in c})==8
def test_modes():
    m=mod(); assert m.mode("GRU")=="v5_uncertainty"; assert m.mode("Mamba")=="v6_uncertainty"
def test_zero_effect():
    m=mod(); rows=[]
    for n,g,i,c in m.build_cases():
        rows.append({"gps_encoder":g,"imu_encoder":i,"camera_encoder":c,"ATE3D":1.,"ATE2D":1.,"ZRMSE":1.})
    assert all(np.isclose(x["delta"],0) for x in m.effects(rows))

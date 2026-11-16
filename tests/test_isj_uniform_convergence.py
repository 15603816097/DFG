from experiments.isj_supplement.run_uniform_convergence import available_cases


def test_core_cases_include_fixed_and_both_frozen_encoders():
    cases = available_cases("core")
    assert [case["name"] for case in cases] == [
        "ISJ_CTRL_FIXED_SECOND_PASS",
        "FINAL_S1_v5_gru",
        "FINAL_S2_v6_all_mamba",
    ]


def test_all_cases_unique_and_permuted_seeds_frozen():
    cases = available_cases("all")
    names = [case["name"] for case in cases]
    assert len(cases) == 15
    assert len(names) == len(set(names))
    for model in ("GRU", "MAMBA"):
        assert f"ISJ_CTRL_{model}_MEAN_U" in names
        for seed in (11, 23, 47):
            assert f"ISJ_CTRL_{model}_PERM_U_SEED_{seed}" in names


def test_controls_is_subset_of_all():
    control_names = {c["name"] for c in available_cases("controls")}
    all_names = {c["name"] for c in available_cases("all")}
    assert len(control_names) == 8
    assert control_names.issubset(all_names)


def test_reliability_only_ablations_in_all_and_ablations():
    reliability = available_cases("reliability")
    assert [case["name"] for case in reliability] == [
        "ABL_V5_GRU_RELIABILITY_ONLY",
        "ABL_V6_MAMBA_RELIABILITY_ONLY",
    ]
    assert {case["type"] for case in reliability} == {"reliability"}
    all_names = {x["name"] for x in available_cases("all")}
    ablations = {x["name"] for x in available_cases("ablations")}
    assert len(ablations) == 10
    assert set(x["name"] for x in reliability).issubset(all_names)
    assert ablations.issubset(all_names)


def test_traditional_adaptive_ablations_are_distinct_and_in_all():
    variants = available_cases("adaptive")
    assert [c["name"] for c in variants] == [
        "ISJ_RESADAPT_GPS",
        "ISJ_RESADAPT_GPS_IMU_CAMERA",
    ]
    assert [tuple(c["sensors"]) for c in variants] == [
        ("gps",),
        ("gps", "imu", "camera"),
    ]
    assert all(c["type"] == "adaptive" for c in variants)
    assert {c["name"] for c in variants}.issubset(
        {c["name"] for c in available_cases("all")}
    )

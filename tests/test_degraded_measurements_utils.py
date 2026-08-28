from src.factor_graph.degraded_measurements import load_clean_reference
def test_import():
    assert callable(load_clean_reference)

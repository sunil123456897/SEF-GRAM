from experiments.memory_keydoor_multiseed import aggregate_rows, parse_seed_list


def test_parse_seed_list():
    assert parse_seed_list("1,2, 3") == (1, 2, 3)


def test_aggregate_rows():
    rows = [
        {"model": "a", "env": "x", "seed": 1, "rollout_obs_mse_avg": 1.0, "has_key_accuracy": 0.5},
        {"model": "a", "env": "x", "seed": 2, "rollout_obs_mse_avg": 3.0, "has_key_accuracy": 1.0},
        {"model": "b", "env": "x", "seed": 1, "rollout_obs_mse_avg": 2.0, "has_key_accuracy": 0.25},
    ]
    out = aggregate_rows(rows)
    by_model = {row["model"]: row for row in out}
    assert by_model["a"]["n_seeds"] == 2
    assert by_model["a"]["rollout_obs_mse_avg_mean"] == 2.0
    assert round(by_model["a"]["rollout_obs_mse_avg_std"], 4) == 1.4142
    assert by_model["a"]["has_key_accuracy_mean"] == 0.75
    assert by_model["b"]["n_seeds"] == 1
    assert by_model["b"]["rollout_obs_mse_avg_std"] == 0.0

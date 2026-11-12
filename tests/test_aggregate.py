import pytest

from aclkv.aggregate import aggregate, parse_workload_name


def _row(seed, cached, policy="B3", users=20, leaks=0):
    return {"workload": f"w_share0.50_mix-default_u{users}_s{seed}", "policy": policy, "kv_gib": "1",
            "concurrency": "16", "cached_pct": str(cached), "evictions": "100",
            "unauthorized_hits": str(leaks), "probe_unauthorized_hits": "0"}


def test_parse_workload_name():
    assert parse_workload_name("w_share0.25_mix-open_u100_s3") == {"share": 0.25, "mix": "open", "users": 100, "seed": 3}
    with pytest.raises(ValueError):
        parse_workload_name("not_a_workload")


def test_aggregate_mean_std_over_seeds():
    rows = [_row(0, 10.0), _row(1, 12.0), _row(2, 14.0)]
    (rec,) = aggregate(rows)
    assert rec["n_seeds"] == 3
    assert rec["cached_pct_mean"] == pytest.approx(12.0)
    assert rec["cached_pct_std"] == pytest.approx(2.0)
    assert (rec["cached_pct_min"], rec["cached_pct_max"]) == (10.0, 14.0)


def test_user_counts_and_policies_are_separate_groups_and_leaks_sum():
    rows = [_row(0, 10, users=20), _row(0, 11, users=50), _row(0, 5, policy="UB", leaks=7), _row(1, 5, policy="UB", leaks=3)]
    out = {(r["users"], r["policy"]): r for r in aggregate(rows)}
    assert set(out) == {(20, "B3"), (50, "B3"), (20, "UB")}
    assert out[(20, "UB")]["unauthorized_hits"] == 10
    assert out[(20, "B3")]["cached_pct_std"] == 0.0   # a single seed has no spread

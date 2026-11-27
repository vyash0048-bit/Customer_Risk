import numpy as np
import pandas as pd

from CustomerRiskPrediction.components.data_transformation import DataTransformation
from CustomerRiskPrediction.components.scorecard import ScorecardExplainer
from woe_iv import calculate_woe_iv, transform_to_woe


def test_calculate_woe_iv_smoothing_and_columns():
    df = pd.DataFrame({
        "feat": ["A", "A", "B", "B", "C", "C"],
        "credit_risk": [0, 0, 1, 1, 0, 1],
    })
    rules = calculate_woe_iv(df, "feat", "credit_risk")

    expected_cols = ["bin", "good_count", "bad_count", "good_distribution", "bad_distribution", "WOE", "IV"]
    assert list(rules.columns) == expected_cols
    assert np.isfinite(rules["WOE"]).all()
    assert (rules["IV"] >= 0).all()


def test_transform_to_woe_handles_known_and_unknown_bins():
    rules = pd.DataFrame({"bin": ["A", "B"], "WOE": [0.5, -0.4]})
    df = pd.DataFrame({"feat": ["A", "B", "UNSEEN"]})

    transformed = transform_to_woe(df, "feat", rules)
    assert transformed["feat_WOE"].tolist() == [0.5, -0.4, 0.0]


def test_open_ended_qcut_maps_extreme_values():
    train = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    val = pd.Series([1, 1000])
    test = pd.Series([0, 50000])

    _, val_binned, test_binned = DataTransformation._qcut_open_ended(train, val, test, q=5)

    assert all(b != "nan" for b in val_binned)
    assert all(b != "nan" for b in test_binned)
    assert val_binned.iloc[0].startswith("(-inf,")
    assert val_binned.iloc[1].endswith("inf]")


def test_interval_matching_supports_open_ended_bounds():
    assert ScorecardExplainer._match_interval("(-inf, 12.0]", -50.0)
    assert ScorecardExplainer._match_interval("(-inf, 12.0]", 12.0)
    assert not ScorecardExplainer._match_interval("(-inf, 12.0]", 12.1)
    assert ScorecardExplainer._match_interval("(30.0, inf]", 72.0)
    assert ScorecardExplainer._match_interval("(30.0, inf]", 50000.0)
    assert ScorecardExplainer._match_interval("<=25", 18.0)
    assert ScorecardExplainer._match_interval("56+", 85.0)

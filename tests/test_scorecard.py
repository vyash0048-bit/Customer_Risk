import os
import joblib
import numpy as np
import pandas as pd

from CustomerRiskPrediction.components.model_trainer import ModelTrainer
from CustomerRiskPrediction.components.scorecard import ScorecardExplainer


def _build_explainer(prior_offset=0.0):
    return ScorecardExplainer(
        model_path=os.path.join("artifacts", "model_trainer", "model.joblib"),
        woe_rules_path=os.path.join("artifacts", "data_transformation", "woe_rules.csv"),
        threshold_path=os.path.join("artifacts", "model_trainer", "lr_threshold.json"),
        prior_offset=prior_offset,
    )


def _sample_customer(**overrides):
    base = {
        "checking_account": "A11",
        "duration": 24,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 2500,
        "savings_account": "A61",
        "employment": "A72",
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "property": "A123",
        "age": 32,
        "other_installment_plans": "A143",
        "housing": "A151",
        "job": "A173",
        "foreign_worker": "A201",
    }
    base.update(overrides)
    return base


def test_prior_odds_shift_formula():
    shift = ModelTrainer.compute_prior_odds_shift(0.30, 0.50)
    expected = np.log(0.30 / 0.70) - np.log(0.50 / 0.50)
    assert np.isclose(shift, expected)
    assert shift < 0


def test_scorecard_points_and_pd_consistency():
    explainer = _build_explainer()
    customer = _sample_customer()
    result = explainer.explain_customer(customer)

    # Verify scorecard formula: Score = round(offset - factor * intercept + sum(contributions))
    unrounded_score = (
        explainer.offset
        - explainer.factor * explainer.intercept
        + sum(c["score_contribution"] for c in result["All Contributions"])
    )
    assert result["Credit Score"] == int(round(unrounded_score))

    # Verify PD matches underlying calibrated sklearn LogisticRegression predict_proba
    model = joblib.load(os.path.join("artifacts", "model_trainer", "model.joblib"))
    x_row = {}
    for woe_col in explainer.features:
        base_feat = woe_col.replace("_WOE", "")
        raw_key = base_feat[:-len("_binned")] if base_feat.endswith("_binned") else base_feat
        woe_val, _ = explainer.map_to_woe(base_feat, customer[raw_key])
        x_row[woe_col] = woe_val

    model_pd = float(model.predict_proba(pd.DataFrame([x_row])[explainer.features])[:, 1][0])
    assert np.isclose(result["Probability of Default"], model_pd, atol=1e-9)

    # Verify base_score at base_odds=50:1 -> PD = 1/51 gives 600
    assert explainer.score_from_pd(1.0 / 51.0) == 600


def test_extreme_numerical_values_do_not_fallback_to_unknown():
    explainer = _build_explainer()
    customer = _sample_customer(age=95, duration=96, credit_amount=75000)
    result = explainer.explain_customer(customer)

    for item in result["All Contributions"]:
        assert item["bin"] != "Unknown"


def test_credit_amount_changes_score_and_drivers():
    explainer = _build_explainer()
    low_amt = explainer.explain_customer(_sample_customer(credit_amount=500))
    high_amt = explainer.explain_customer(_sample_customer(credit_amount=25000))

    low_contrib = next(c for c in low_amt["All Contributions"] if c["feature"] == "credit_amount_binned")
    high_contrib = next(c for c in high_amt["All Contributions"] if c["feature"] == "credit_amount_binned")

    assert low_contrib["bin"] != high_contrib["bin"]
    assert low_contrib["score_contribution"] != high_contrib["score_contribution"]
    assert low_amt["Probability of Default"] != high_amt["Probability of Default"]


def test_decision_and_grade_monotonic_with_calibrated_pd():
    explainer = _build_explainer()
    assert explainer.assign_decision(explainer.approve_pd_max - 0.01) == "APPROVE"
    assert explainer.assign_decision(explainer.approve_pd_max) == "MANUAL REVIEW"
    assert explainer.assign_decision(explainer.reject_pd_min) == "REJECT"

    grades = [
        explainer.assign_risk_grade(p)
        for p in [0.02, explainer.approve_pd_max * 0.8, 0.25, explainer.reject_pd_min - 0.05, 0.70]
    ]
    assert grades == ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]

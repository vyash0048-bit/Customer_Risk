import json
import pytest

from dashboard.app import PredictionAudit, create_app, db


@pytest.fixture()
def client():
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_get_dashboard_exposes_active_features(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Check key fields and extended purpose codes are present
    for field_name in ["age", "duration", "credit_amount", "checking_account", "purpose", "job"]:
        assert f'name="{field_name}"' in html
    assert 'value="A410"' in html


def test_get_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert "Model Performance Metrics" in resp.get_data(as_text=True)


def test_api_predict_success_and_audit_log(client):
    payload = {
        "age": 45,
        "credit_amount": 2000,
        "duration": 12,
        "purpose": "A43",
        "checking_account": "A14",
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()

    assert body["status"] == "success"
    assert body["db_status"] == "success"
    assert isinstance(body["credit_score"], int)
    assert 0.0 < body["probability_of_default"] < 1.0
    assert body["decision"] in {"APPROVE", "MANUAL REVIEW", "REJECT"}
    assert body["features_used"]["purpose"] == "A43"

    with client.application.app_context():
        records = PredictionAudit.query.all()
        assert len(records) == 1
        saved_features = json.loads(records[0].features_json)
        assert saved_features["credit_amount"] == 2000.0
        assert saved_features["purpose"] == "A43"


@pytest.mark.parametrize("bad_payload", [
    {},
    {"age": 12, "duration": 12, "credit_amount": 2000},
    {"age": 30, "duration": 0, "credit_amount": 2000},
    {"age": 30, "duration": 12, "credit_amount": -100},
    {"age": "not-a-number", "duration": 12, "credit_amount": 2000},
    {"age": 30, "duration": 12, "credit_amount": 2000, "purpose": "INVALID_CODE"},
])
def test_api_predict_validation_errors_return_400(client, bad_payload):
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["status"] == "error"


def test_form_post_validation_and_success(client):
    # Invalid age returns 400 with inline validation error
    bad_resp = client.post("/", data={"age": "15", "duration": "12", "credit_amount": "2000"})
    assert bad_resp.status_code == 400
    assert "Validation Error" in bad_resp.get_data(as_text=True)

    # Complete valid form returns 200 and logs audit entry
    explainer = client.application.extensions["scorecard_explainer"]
    form_data = {
        "age": "40",
        "duration": "12",
        "credit_amount": "1500",
        "checking_account": "A14",
        "savings_account": "A65",
        "employment": "A75",
        "credit_history": "A34",
        "purpose": "A43",
        "housing": "A152",
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "property": "A121",
        "other_installment_plans": "A143",
        "job": "A173",
        "foreign_worker": "A201",
    }
    active_set = set(explainer.get_active_base_features()) | {"age", "duration", "credit_amount"}
    filtered_form = {k: v for k, v in form_data.items() if k in active_set}

    ok_resp = client.post("/", data=filtered_form)
    assert ok_resp.status_code == 200
    assert "Score Contribution by Feature" in ok_resp.get_data(as_text=True)

from datetime import datetime, timezone
import json
import math
import os
import urllib.parse

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

from CustomerRiskPrediction.components.scorecard import ScorecardExplainer
from CustomerRiskPrediction.logger import logger

db = SQLAlchemy()

FEATURE_SCHEMA = {
    "age": {
        "label": "Age (Years)",
        "type": "number",
        "min": 18,
        "max": 120,
        "step": "1",
        "default": 35,
        "placeholder": "e.g. 35",
    },
    "duration": {
        "label": "Loan Duration (Months)",
        "type": "number",
        "min": 1,
        "max": 120,
        "step": "1",
        "default": 24,
        "placeholder": "e.g. 24",
    },
    "credit_amount": {
        "label": "Credit Amount (DM)",
        "type": "number",
        "min": 1,
        "max": 1000000,
        "step": "1",
        "default": 2000,
        "placeholder": "e.g. 2000",
    },
    "checking_account": {
        "label": "Checking Account",
        "type": "select",
        "default": "A14",
        "options": [
            ("A11", "< 0 DM"),
            ("A12", "0 <= x < 200 DM"),
            ("A13", ">= 200 DM / salary assignments"),
            ("A14", "No checking account"),
        ],
    },
    "savings_account": {
        "label": "Savings Account",
        "type": "select",
        "default": "A65",
        "options": [
            ("A61", "< 100 DM"),
            ("A62", "100 <= x < 500 DM"),
            ("A63", "500 <= x < 1000 DM"),
            ("A64", ">= 1000 DM"),
            ("A65", "Unknown / no savings account"),
        ],
    },
    "employment": {
        "label": "Employment Duration",
        "type": "select",
        "default": "A73",
        "options": [
            ("A71", "Unemployed"),
            ("A72", "< 1 year"),
            ("A73", "1 <= x < 4 years"),
            ("A74", "4 <= x < 7 years"),
            ("A75", ">= 7 years"),
        ],
    },
    "credit_history": {
        "label": "Credit History",
        "type": "select",
        "default": "A32",
        "options": [
            ("A30", "No credits taken / all paid back"),
            ("A31", "All credits at this bank paid back duly"),
            ("A32", "Existing credits paid back duly till now"),
            ("A33", "Delay in paying off in the past"),
            ("A34", "Critical account / other credits existing"),
        ],
    },
    "purpose": {
        "label": "Loan Purpose",
        "type": "select",
        "default": "A43",
        "options": [
            ("A40", "Car (new)"),
            ("A41", "Car (used)"),
            ("A42", "Furniture / equipment"),
            ("A43", "Radio / television"),
            ("A44", "Domestic appliances"),
            ("A45", "Repairs"),
            ("A46", "Education"),
            ("A48", "Retraining"),
            ("A49", "Business"),
            ("A410", "Others"),
        ],
    },
    "housing": {
        "label": "Housing",
        "type": "select",
        "default": "A152",
        "options": [
            ("A151", "Rent"),
            ("A152", "Own"),
            ("A153", "For free"),
        ],
    },
    "personal_status_sex": {
        "label": "Personal Status & Sex",
        "type": "select",
        "default": "A93",
        "options": [
            ("A91", "Male: divorced / separated"),
            ("A92", "Female: divorced / separated / married"),
            ("A93", "Male: single"),
            ("A94", "Male: married / widowed"),
        ],
    },
    "other_debtors": {
        "label": "Other Debtors / Guarantors",
        "type": "select",
        "default": "A101",
        "options": [
            ("A101", "None"),
            ("A102", "Co-applicant"),
            ("A103", "Guarantor"),
        ],
    },
    "property": {
        "label": "Property",
        "type": "select",
        "default": "A121",
        "options": [
            ("A121", "Real estate"),
            ("A122", "Building society savings / life insurance"),
            ("A123", "Car or other"),
            ("A124", "Unknown / no property"),
        ],
    },
    "other_installment_plans": {
        "label": "Other Installment Plans",
        "type": "select",
        "default": "A143",
        "options": [
            ("A141", "Bank"),
            ("A142", "Stores"),
            ("A143", "None"),
        ],
    },
    "job": {
        "label": "Job / Occupation",
        "type": "select",
        "default": "A173",
        "options": [
            ("A171", "Unemployed / unskilled (non-resident)"),
            ("A172", "Unskilled (resident)"),
            ("A173", "Skilled employee / official"),
            ("A174", "Management / self-employed / highly qualified"),
        ],
    },
    "foreign_worker": {
        "label": "Foreign Worker",
        "type": "select",
        "default": "A201",
        "options": [
            ("A201", "Yes"),
            ("A202", "No"),
        ],
    },
}


class PredictionAudit(db.Model):
    __tablename__ = 'prediction_audit'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    age = db.Column(db.Float)
    credit_amount = db.Column(db.Float)
    duration = db.Column(db.Float)
    credit_score = db.Column(db.Integer)
    prob_default = db.Column(db.Float)
    decision = db.Column(db.String(50))
    risk_grade = db.Column(db.String(50))
    features_json = db.Column(db.Text)


def init_db(app_instance: Flask) -> bool:
    """Create database tables and ensure schema columns exist without crashing app import."""
    try:
        with app_instance.app_context():
            db.create_all()
            if db.engine.dialect.name == "postgresql":
                db.session.execute(
                    text("ALTER TABLE prediction_audit ADD COLUMN IF NOT EXISTS features_json TEXT")
                )
                db.session.commit()
        return True
    except Exception as exc:
        logger.warning(f"Database initialization skipped or failed: {exc}")
        return False


def _build_active_fields(explainer: ScorecardExplainer):
    """Build ordered UI/validation field definitions matching features retained in woe_rules.csv."""
    active_raw = set(explainer.get_active_base_features()) | {"age", "duration", "credit_amount"}
    fields = []
    for name, meta in FEATURE_SCHEMA.items():
        if name not in active_raw:
            continue
        field = dict(meta)
        field["name"] = name
        if field["type"] == "select":
            valid_bins = set(explainer.get_valid_categories(name))
            field["options"] = [opt for opt in meta["options"] if not valid_bins or opt[0] in valid_bins]
        fields.append(field)
    return fields


def _validate_and_extract(payload: dict, active_fields: list, explainer: ScorecardExplainer, allow_defaults: bool):
    """Validate numeric bounds and categorical codes against active model features."""
    errors = []
    cleaned = {}

    for field in active_fields:
        name = field["name"]
        raw_val = payload.get(name)

        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            if allow_defaults:
                cleaned[name] = field["default"]
                continue
            errors.append(f"Field '{field['label']}' ({name}) is required.")
            continue

        if field["type"] == "number":
            if isinstance(raw_val, bool):
                errors.append(f"Field '{name}' must be a valid number.")
                continue
            try:
                num_val = float(raw_val)
            except (TypeError, ValueError):
                errors.append(f"Field '{name}' must be a valid number.")
                continue
            if not math.isfinite(num_val):
                errors.append(f"Field '{name}' must be finite.")
                continue
            if name == "age" and num_val < 18:
                errors.append("Field 'age' must be >= 18.")
                continue
            if name in {"duration", "credit_amount"} and num_val <= 0:
                errors.append(f"Field '{name}' must be > 0.")
                continue
            cleaned[name] = num_val
        else:
            code_val = str(raw_val).strip()
            valid_codes = {opt[0] for opt in field["options"]}
            if valid_codes and code_val not in valid_codes:
                errors.append(
                    f"Invalid category '{code_val}' for '{name}'. Expected one of: {sorted(valid_codes)}."
                )
                continue
            cleaned[name] = code_val

    return cleaned, errors


def _decision_css_class(decision: str) -> str:
    if decision == "APPROVE":
        return "approve"
    if decision == "REJECT":
        return "reject"
    return "review"


def create_app(config_overrides=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    load_dotenv(os.path.join(base_dir, '.env'))

    flask_app = Flask(__name__)

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", "postgres"))
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "credit_risk")
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    flask_app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    if config_overrides:
        flask_app.config.update(config_overrides)

    db.init_app(flask_app)

    model_path = os.path.join(base_dir, "artifacts", "model_trainer", "model.joblib")
    woe_rules_path = os.path.join(base_dir, "artifacts", "data_transformation", "woe_rules.csv")
    threshold_path = os.path.join(base_dir, "artifacts", "model_trainer", "lr_threshold.json")

    explainer = ScorecardExplainer(
        model_path=model_path,
        woe_rules_path=woe_rules_path,
        threshold_path=threshold_path,
    )
    active_fields = _build_active_fields(explainer)
    flask_app.extensions["scorecard_explainer"] = explainer

    if not flask_app.config.get("TESTING", False):
        init_db(flask_app)

    def _log_audit(customer_data: dict, result: dict) -> str:
        try:
            audit_record = PredictionAudit(
                age=customer_data['age'],
                credit_amount=customer_data['credit_amount'],
                duration=customer_data['duration'],
                credit_score=result['Credit Score'],
                prob_default=float(result['Probability of Default']),
                decision=result['Decision'],
                risk_grade=result['Risk Grade'],
                features_json=json.dumps(customer_data),
            )
            db.session.add(audit_record)
            db.session.commit()
            return "success"
        except Exception as exc:
            db.session.rollback()
            logger.error(f"Error saving audit record to database: {exc}")
            return f"error: {str(exc)}"

    @flask_app.route('/', methods=['GET', 'POST'])
    def index():
        result = None
        errors = []
        status_code = 200
        form_data = request.form if request.method == 'POST' else {}

        if request.method == 'POST':
            customer_data, errors = _validate_and_extract(
                request.form.to_dict(), active_fields, explainer, allow_defaults=False
            )
            if errors:
                status_code = 400
            else:
                result = explainer.explain_customer(customer_data)
                result['Probability of Default Formatted'] = f"{result['Probability of Default'] * 100:.1f}%"
                score = result['Credit Score']
                result['Score_Percent'] = max(0, min(100, (score - 400) / 200 * 100))
                result['Decision_Class'] = _decision_css_class(result['Decision'])
                result['db_status'] = _log_audit(customer_data, result)

        return render_template(
            'index.html',
            result=result,
            errors=errors,
            form=form_data,
            fields=active_fields,
        ), status_code

    @flask_app.route('/api/v1/predict', methods=['POST'])
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return jsonify({"status": "error", "message": "Valid non-empty JSON object is required."}), 400

        customer_data, errors = _validate_and_extract(
            data, active_fields, explainer, allow_defaults=True
        )
        if errors:
            return jsonify({
                "status": "error",
                "message": "Input validation failed.",
                "errors": errors,
            }), 400

        try:
            result = explainer.explain_customer(customer_data)
            db_status = _log_audit(customer_data, result)
            return jsonify({
                "status": "success",
                "db_status": db_status,
                "credit_score": result['Credit Score'],
                "probability_of_default": result['Probability of Default'],
                "risk_grade": result['Risk Grade'],
                "decision": result['Decision'],
                "decision_thresholds": {
                    "approve_pd_max": explainer.approve_pd_max,
                    "reject_pd_min": explainer.reject_pd_min,
                    "approve_score_min": explainer.approve_score_min,
                    "reject_score_max": explainer.reject_score_max,
                },
                "features_used": customer_data,
            }), 200
        except Exception as exc:
            logger.exception(f"Unexpected error in /api/v1/predict: {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @flask_app.route('/model-info')
    def model_info():
        metrics_path = os.path.join(base_dir, "artifacts", "model_evaluation", "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        return render_template('model_info.html', metrics=metrics)

    return flask_app


app = create_app()

if __name__ == '__main__':
    init_db(app)
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)

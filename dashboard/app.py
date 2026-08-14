from flask import Flask, render_template, request, jsonify
import sys
import os

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.CustomerRiskPrediction.components.scorecard import ScorecardExplainer

app = Flask(__name__)

from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Database Configuration
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "credit_risk")

import urllib.parse
DB_PASSWORD_ENCODED = urllib.parse.quote_plus(DB_PASSWORD)

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class PredictionAudit(db.Model):
    __tablename__ = 'prediction_audit'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Float)
    credit_amount = db.Column(db.Float)
    duration = db.Column(db.Float)
    credit_score = db.Column(db.Integer)
    prob_default = db.Column(db.Float)
    decision = db.Column(db.String(50))
    risk_grade = db.Column(db.String(50))

# Initialize explainer (paths are relative to the dashboard directory when running from it, 
# so we assume running from the root directory or adjust paths accordingly)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(base_dir, "artifacts", "model_trainer", "model.joblib")
woe_rules_path = os.path.join(base_dir, "artifacts", "data_transformation", "woe_rules.csv")

explainer = ScorecardExplainer(model_path=model_path, woe_rules_path=woe_rules_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Extract data from the form, using defaults for unlisted but required model features
        customer_data = {
            "age": float(request.form.get("age", 30)),
            "employment": request.form.get("employment"),
            "housing": request.form.get("housing"),
            "credit_amount": float(request.form.get("credit_amount", 2000)),
            "duration": float(request.form.get("duration", 24)),
            "checking_account": request.form.get("checking_account"),
            "savings_account": request.form.get("savings_account"),
            "credit_history": request.form.get("credit_history"),
            "purpose": request.form.get("purpose"),
            "personal_status_sex": request.form.get("personal_status_sex", "A93"), # Default: male single
            "other_debtors": request.form.get("other_debtors", "A101"), # Default: none
            "property": request.form.get("property", "A121"), # Default: real estate
            "other_installment_plans": request.form.get("other_installment_plans", "A143"), # Default: none
            "job": request.form.get("job", "A173"), # Default: skilled
            "foreign_worker": request.form.get("foreign_worker", "A201") # Default: yes
        }
        
        # Calculate scorecard
        result = explainer.explain_customer(customer_data)
        
        # Format Probability of Default
        result['Probability of Default Formatted'] = f"{result['Probability of Default'] * 100:.1f}%"
        
        # Determine Decision Rule
        score = result['Credit Score']
        print(f"DEBUG: Score is {score}, type {type(score)}")
        
        # Calculate percentage for UI gauge (assuming realistic bounds of 400 to 600)
        score_percent = max(0, min(100, (score - 400) / 200 * 100))
        result['Score_Percent'] = score_percent
        
        if score < 500:
            result['Decision'] = "REJECT"
            result['Decision_Class'] = "reject"
        elif 500 <= score < 540:
            result['Decision'] = "MANUAL REVIEW"
            result['Decision_Class'] = "review"
        else:
            result['Decision'] = "APPROVE"
            result['Decision_Class'] = "approve"
        print(f"DEBUG: Final Decision is {result['Decision']}")
        
        # Save to PostgreSQL Audit Log
        try:
            audit_record = PredictionAudit(
                age=customer_data['age'],
                credit_amount=customer_data['credit_amount'],
                duration=customer_data['duration'],
                credit_score=result['Credit Score'],
                prob_default=float(result['Probability of Default']),
                decision=result['Decision'],
                risk_grade=result['Risk Grade']
            )
            db.session.add(audit_record)
            db.session.commit()
            result['db_status'] = "success"
            print("Successfully saved prediction to PostgreSQL database.")
        except Exception as e:
            print(f"Error saving to database: {e}")
            db.session.rollback()
            result['db_status'] = f"error: {str(e)}"
            
    form_data = request.form if request.method == 'POST' else {}
    return render_template('index.html', result=result, form=form_data)

@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        customer_data = {
            "age": float(data.get("age", 30)),
            "employment": data.get("employment", "A73"),
            "housing": data.get("housing", "A152"),
            "credit_amount": float(data.get("credit_amount", 2000)),
            "duration": float(data.get("duration", 24)),
            "checking_account": data.get("checking_account", "A14"),
            "savings_account": data.get("savings_account", "A65"),
            "credit_history": data.get("credit_history", "A32"),
            "purpose": data.get("purpose", "A43"),
            "personal_status_sex": data.get("personal_status_sex", "A93"),
            "other_debtors": data.get("other_debtors", "A101"),
            "property": data.get("property", "A121"),
            "other_installment_plans": data.get("other_installment_plans", "A143"),
            "job": data.get("job", "A173"),
            "foreign_worker": data.get("foreign_worker", "A201")
        }
        
        result = explainer.explain_customer(customer_data)
        score = result['Credit Score']
        
        if score < 500:
            decision = "REJECT"
        elif 500 <= score < 540:
            decision = "MANUAL REVIEW"
        else:
            decision = "APPROVE"
            
        # Log to Database
        try:
            audit_record = PredictionAudit(
                age=customer_data['age'],
                credit_amount=customer_data['credit_amount'],
                duration=customer_data['duration'],
                credit_score=score,
                prob_default=float(result['Probability of Default']),
                decision=decision,
                risk_grade=result['Risk Grade']
            )
            db.session.add(audit_record)
            db.session.commit()
            db_status = "success"
        except Exception as e:
            db.session.rollback()
            db_status = f"error: {str(e)}"
            
        return jsonify({
            "status": "success",
            "db_status": db_status,
            "credit_score": score,
            "probability_of_default": result['Probability of Default'],
            "risk_grade": result['Risk Grade'],
            "decision": decision,
            "features_used": customer_data
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/model-info')
def model_info():
    metrics_path = os.path.join(base_dir, "artifacts", "model_evaluation", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return render_template('model_info.html', metrics=metrics)

# Create database tables before starting the app
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, port=5000)

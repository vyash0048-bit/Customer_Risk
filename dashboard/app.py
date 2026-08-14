from flask import Flask, render_template, request
import sys
import os

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.CustomerRiskPrediction.components.scorecard import ScorecardExplainer

app = Flask(__name__)

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
            
    form_data = request.form if request.method == 'POST' else {}
    return render_template('index.html', result=result, form=form_data)

@app.route('/model-info')
def model_info():
    metrics_path = os.path.join(base_dir, "artifacts", "model_evaluation", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return render_template('model_info.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import sys
import os
import pandas as pd
import json

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from CustomerRiskPrediction.components.scorecard import ScorecardBuilder

def generate_report():
    print("Loading model and data...")
    sb = ScorecardBuilder(
        model_path='artifacts/model_trainer/model.joblib',
        woe_rules_path='artifacts/data_transformation/woe_rules.csv'
    )
    
    test_data = pd.read_csv('artifacts/data_transformation/test.csv')
    woe_columns = [col for col in test_data.columns if col.endswith('_WOE')]
    
    X_test = test_data[woe_columns]
    y_test = test_data['credit_risk']
    
    y_pred_proba = sb.model.predict_proba(X_test)[:, 1]
    
    print("Generating Scorecard Table...")
    scorecard_df, base_points = sb.generate_scorecard()
    scorecard_df.to_csv('artifacts/model_evaluation/scorecard.csv', index=False)
    
    print("Calculating Reliability Table...")
    reliability_table, brier = sb.calculate_reliability_table(y_test, y_pred_proba)
    reliability_table.to_csv('artifacts/model_evaluation/reliability.csv', index=False)
    
    print("Generating Customer Explanations...")
    
    # Pick a few sample customers
    customer_1_raw = test_data.iloc[0].drop(woe_columns).to_dict()
    customer_1_woe = test_data.iloc[0][woe_columns]
    
    explanation_1 = sb.explain_customer(customer_1_raw, customer_1_woe)
    
    customer_2_raw = test_data.iloc[5].drop(woe_columns).to_dict()
    customer_2_woe = test_data.iloc[5][woe_columns]
    
    explanation_2 = sb.explain_customer(customer_2_raw, customer_2_woe)
    
    # Generate Markdown Report
    report_path = 'research/Scorecard_Report.md'
    with open(report_path, 'w') as f:
        f.write("# Model Calibration & Scorecard Report\n\n")
        
        f.write("## 1. Reliability & Calibration\n")
        f.write(f"**Brier Score:** {brier:.4f}\n\n")
        f.write("### Reliability Table\n")
        f.write(reliability_table.to_markdown(index=False) + "\n\n")
        
        f.write("## 2. Scorecard Construction\n")
        f.write(f"**Illustrative Base Score:** {sb.base_score}\n")
        f.write(f"**Illustrative Base Odds:** {sb.base_odds}:1\n")
        f.write(f"**Points to Double the Odds (PDO):** {sb.pdo}\n\n")
        
        f.write(f"**Calculated Offset:** {sb.offset:.2f}\n")
        f.write(f"**Calculated Factor:** {sb.factor:.2f}\n")
        f.write(f"**Base Points (Offset + Intercept):** {base_points}\n\n")
        
        f.write("### Scorecard Sample (Top 10)\n")
        f.write(scorecard_df.head(10).to_markdown(index=False) + "\n\n")
        
        f.write("## 3. Score Interpretation Rules\n")
        f.write("- Score >= 750: Very Low Risk\n")
        f.write("- 700-749: Low Risk\n")
        f.write("- 650-699: Moderate Risk\n")
        f.write("- 600-649: High Risk\n")
        f.write("- < 600: Very High Risk\n\n")
        
        f.write("## 4. Customer-Level Explanations\n")
        f.write("### Example 1\n")
        f.write("```json\n")
        f.write(json.dumps(explanation_1, indent=2) + "\n")
        f.write("```\n")
        
        f.write("### Example 2\n")
        f.write("```json\n")
        f.write(json.dumps(explanation_2, indent=2) + "\n")
        f.write("```\n")
        
    print(f"Report successfully saved to {report_path}")

if __name__ == '__main__':
    generate_report()

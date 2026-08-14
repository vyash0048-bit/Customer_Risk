import pandas as pd
import numpy as np
import joblib
import os

class ScorecardExplainer:
    def __init__(self, model_path, woe_rules_path, base_score=600, base_odds=50, pdo=20):
        self.model = joblib.load(model_path)
        self.woe_rules = pd.read_csv(woe_rules_path)
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        
        # Get intercept
        if hasattr(self.model, 'intercept_'):
            self.intercept = self.model.intercept_[0]
        else:
            self.intercept = 0.0
            
        # Get coefficients
        if hasattr(self.model, 'feature_names_in_'):
            self.features = list(self.model.feature_names_in_)
        else:
            raise ValueError("Model must have feature_names_in_")
            
        self.coefs = {f: c for f, c in zip(self.features, self.model.coef_[0])}
        
    def map_to_woe(self, feature_name, value):
        rules = self.woe_rules[self.woe_rules['feature'] == feature_name]
        
        if feature_name == 'age_binned':
            for b in rules['bin'].values:
                if '+' in str(b):
                    lower = float(b.replace('+', ''))
                    if value >= lower: return float(rules[rules['bin'] == b]['WOE'].values[0]), b
                elif '-' in str(b):
                    parts = b.split('-')
                    lower, upper = float(parts[0]), float(parts[1])
                    if lower <= value <= upper: return float(rules[rules['bin'] == b]['WOE'].values[0]), b
        elif feature_name == 'duration_binned':
            for b in rules['bin'].values:
                clean_b = str(b).replace('(', '').replace('[', '').replace(']', '').replace(')', '')
                parts = clean_b.split(',')
                if len(parts) == 2:
                    lower, upper = float(parts[0]), float(parts[1])
                    if (lower < value <= upper) or (str(b).startswith('[') and lower <= value <= upper):
                        return float(rules[rules['bin'] == b]['WOE'].values[0]), b
        else:
            match = rules[rules['bin'] == str(value)]
            if not match.empty:
                return float(match['WOE'].values[0]), str(value)
            
        return 0.0, "Unknown"
        
    def explain_customer(self, customer_dict):
        score = self.offset - self.factor * self.intercept
        contributions = []
        
        log_odds_sum = self.intercept
        
        for woe_col in self.features:
            base_feature = woe_col.replace('_WOE', '')
            
            if base_feature == 'age_binned':
                val = customer_dict.get('age')
                woe_val, bin_val = self.map_to_woe('age_binned', val)
            elif base_feature == 'duration_binned':
                val = customer_dict.get('duration')
                woe_val, bin_val = self.map_to_woe('duration_binned', val)
            else:
                val = customer_dict.get(base_feature)
                woe_val, bin_val = self.map_to_woe(base_feature, val)
                
            coef = self.coefs[woe_col]
            
            # Log odds contribution (Logistic regression is trained to predict P(Bad))
            log_odds_contrib = coef * woe_val
            log_odds_sum += log_odds_contrib
            
            # Score contribution: Score = Offset - Factor * (w^T * x + b)
            # A negative log_odds_contrib (decreasing risk) increases the score.
            contrib = - self.factor * log_odds_contrib
            score += contrib
            
            contributions.append({
                'feature': base_feature,
                'original_value': val,
                'bin': bin_val,
                'woe': woe_val,
                'coef': coef,
                'score_contribution': contrib,
                'log_odds_contribution': log_odds_contrib
            })
            
        prob = 1 / (1 + np.exp(-log_odds_sum))
        final_score = int(round(score))
        
        if final_score >= 560:
            grade = "Very Low Risk"
        elif 530 <= final_score < 560:
            grade = "Low Risk"
        elif 500 <= final_score < 530:
            grade = "Moderate Risk"
        elif 460 <= final_score < 500:
            grade = "High Risk"
        else:
            grade = "Very High Risk"
            
        sorted_contribs = sorted(contributions, key=lambda x: x['score_contribution'])
        
        risk_drivers = sorted_contribs[:5]
        positive_drivers = sorted(contributions, key=lambda x: x['score_contribution'], reverse=True)[:5]
        
        result = {
            'Credit Score': final_score,
            'Probability of Default': prob,
            'Risk Grade': grade,
            'Risk Drivers': risk_drivers,
            'Positive Drivers': positive_drivers,
            'All Contributions': contributions
        }
        
        return result

    def format_explanation(self, result):
        lines = []
        lines.append(f"Credit Score: {result['Credit Score']}")
        lines.append(f"Probability of Default: {result['Probability of Default'] * 100:.1f}%")
        lines.append(f"Risk Grade: {result['Risk Grade'].upper()}\n")
        
        lines.append("Risk Drivers:")
        for i, driver in enumerate(result['Risk Drivers'], 1):
            if driver['score_contribution'] < 0:
                lines.append(f"{i}. {driver['feature']} (Value: {driver['original_value']}, Bin: {driver['bin']}) -> {driver['score_contribution']:.1f} points")
            
        lines.append("\nPositive Drivers:")
        for i, driver in enumerate(result['Positive Drivers'], 1):
            if driver['score_contribution'] > 0:
                lines.append(f"{i}. {driver['feature']} (Value: {driver['original_value']}, Bin: {driver['bin']}) -> +{driver['score_contribution']:.1f} points")
                
        lines.append("\n* Note: These are model-derived risk factors, not causal claims. Score boundaries and parameters (Base 600, PDO 20) are illustrative.")
        return "\n".join(lines)

if __name__ == "__main__":
    # Test execution
    explainer = ScorecardExplainer(
        model_path=os.path.join("artifacts", "model_trainer", "model.joblib"),
        woe_rules_path=os.path.join("artifacts", "data_transformation", "woe_rules.csv")
    )
    
    # Illustrative customer
    sample_customer = {
        "checking_account": "A11",
        "duration": 12,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 1295,
        "savings_account": "A61",
        "employment": "A72",
        "personal_status_sex": "A92",
        "other_debtors": "A101",
        "property": "A123",
        "age": 25,
        "other_installment_plans": "A143",
        "housing": "A151",
        "job": "A173",
        "foreign_worker": "A201"
    }
    
    result = explainer.explain_customer(sample_customer)
    print(explainer.format_explanation(result))

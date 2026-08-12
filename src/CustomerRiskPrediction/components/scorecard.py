import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

class ScorecardBuilder:
    def __init__(self, model_path, woe_rules_path):
        self.model = joblib.load(model_path)
        self.woe_rules = pd.read_csv(woe_rules_path)
        
        # We need the feature names that the model was trained on
        # Assuming model.feature_names_in_ is available
        self.features = self.model.feature_names_in_
        
        # Extract coefficients
        self.intercept = self.model.intercept_[0]
        self.coef = self.model.coef_[0]
        self.coef_dict = dict(zip(self.features, self.coef))
        
        # Scorecard parameters
        self.pdo = 20
        self.base_score = 600
        self.base_odds = 50  # 50:1 Good to Bad odds
        
        # Calculate Factor and Offset
        # Note: the model predicts Default (1) which is "Bad". 
        # Standard scorecard: Score = Offset - Factor * ln(Odds of Bad)
        # Or if we want higher score for lower risk (Better):
        # Probability of Default = P(Bad)
        # Log-odds of Bad = intercept + sum(coef * woe)
        # We want higher score = Good (Low Risk).
        # Score = Offset - Factor * Log-odds(Bad)
        # Log-odds(Bad) = - Log-odds(Good)
        # Factor = PDO / ln(2)
        
        self.factor = self.pdo / np.log(2)
        
        # Base Odds = 50:1 means Odds(Good/Bad) = 50.
        # Log-odds(Bad) = ln(1/50) = -ln(50)
        # Base Score = Offset - Factor * ln(1/50) => Offset = Base Score + Factor * ln(1/50)
        
        # Wait, usually Base Odds is defined as Good:Bad odds.
        # Odds(Good) = 50. ln(Odds(Good)) = ln(50)
        # Score = Offset + Factor * ln(Odds(Good))
        # 600 = Offset + (20/ln(2)) * ln(50)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        
    def generate_scorecard(self):
        """Generates the scorecard points for every bin of every feature."""
        scorecard = []
        
        # The base points include the intercept scaled by the factor + offset
        base_points = self.offset - self.factor * self.intercept
        
        for index, row in self.woe_rules.iterrows():
            feature = row['feature']
            woe_col = f"{feature}_WOE"
            if woe_col in self.coef_dict:
                woe = row['WOE']
                coef = self.coef_dict[woe_col]
                
                # Point contribution = - Factor * (Coefficient * WOE)
                # Why minus? Because coef * WOE is the log-odds of BAD.
                # Score = Offset - Factor * (intercept + sum(coef * WOE))
                points = -self.factor * coef * woe
                
                scorecard.append({
                    'Feature': feature,
                    'Bin': row['bin'],
                    'WOE': woe,
                    'Coefficient': coef,
                    'Points': round(points)
                })
                
        return pd.DataFrame(scorecard), round(base_points)

    def calculate_reliability_table(self, y_true, y_pred_proba, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='quantile')
        brier = brier_score_loss(y_true, y_pred_proba)
        
        reliability_table = pd.DataFrame({
            'Predicted PD': prob_pred,
            'Actual Default Rate': prob_true
        })
        
        # Optional: formatting
        reliability_table['Predicted PD'] = reliability_table['Predicted PD'].apply(lambda x: f"{x*100:.2f}%")
        reliability_table['Actual Default Rate'] = reliability_table['Actual Default Rate'].apply(lambda x: f"{x*100:.2f}%")
        
        return reliability_table, brier

    def predict_score(self, X_woe):
        """Calculate the credit score for a given row or dataframe of WOE features."""
        # Log-odds of Bad
        log_odds_bad = self.model.decision_function(X_woe)
        
        # Score = Offset - Factor * Log-odds_bad
        score = self.offset - self.factor * log_odds_bad
        return score
        
    def get_risk_grade(self, score):
        if score >= 750:
            return "Very Low Risk"
        elif 700 <= score <= 749:
            return "Low Risk"
        elif 650 <= score <= 699:
            return "Moderate Risk"
        elif 600 <= score <= 649:
            return "High Risk"
        else:
            return "Very High Risk"

    def explain_customer(self, customer_raw, customer_woe):
        """
        Explain the scorecard for a specific customer.
        customer_raw: Series or dict of raw features
        customer_woe: Series or dict of WOE features
        """
        # Calculate predicted probability
        # Reshape for single prediction if necessary
        X = pd.DataFrame([customer_woe])
        pd_prob = self.model.predict_proba(X)[0][1]
        
        score = self.predict_score(X)[0]
        risk_grade = self.get_risk_grade(score)
        
        contributions = []
        for feature in self.features:
            coef = self.coef_dict[feature]
            woe_val = customer_woe[feature]
            raw_feature = feature.replace('_WOE', '')
            raw_val = customer_raw.get(raw_feature, "Unknown")
            
            # Points for this feature
            points = -self.factor * coef * woe_val
            contributions.append({
                'Feature': raw_feature,
                'Value': raw_val,
                'Points': points
            })
            
        contributions = sorted(contributions, key=lambda x: x['Points'])
        
        # Negative points increase risk (Risk Drivers)
        risk_drivers = [c for c in contributions if c['Points'] < 0]
        # Positive points decrease risk (Positive Drivers)
        positive_drivers = sorted([c for c in contributions if c['Points'] > 0], key=lambda x: x['Points'], reverse=True)
        
        result = {
            "Credit Score": round(score),
            "Probability of Default": f"{pd_prob*100:.1f}%",
            "Risk Grade": risk_grade,
            "Risk Drivers": risk_drivers[:4],
            "Positive Drivers": positive_drivers[:4]
        }
        
        return result

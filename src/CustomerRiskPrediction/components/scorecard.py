import json
import os
import joblib
import numpy as np
import pandas as pd


class ScorecardExplainer:
    def __init__(
        self,
        model_path,
        woe_rules_path,
        base_score=600,
        base_odds=50,
        pdo=20,
        threshold_path=None,
        lr_threshold=None,
        prior_offset=0.0,
    ):
        self.model = joblib.load(model_path)
        self.woe_rules = pd.read_csv(woe_rules_path)
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.prior_offset = float(prior_offset)

        if lr_threshold is not None:
            self.lr_threshold = float(lr_threshold)
        elif threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.lr_threshold = float(json.load(f).get('lr_threshold', 0.25))
        else:
            default_thresh_path = os.path.join(os.path.dirname(model_path), "lr_threshold.json")
            if os.path.exists(default_thresh_path):
                with open(default_thresh_path, 'r') as f:
                    self.lr_threshold = float(json.load(f).get('lr_threshold', 0.25))
            else:
                self.lr_threshold = 0.25

        # Calibrated PD thresholds for 3-way underwriting decision:
        # APPROVE: PD < approve_pd_max (anchored to F2 threshold)
        # MANUAL REVIEW: approve_pd_max <= PD < reject_pd_min
        # REJECT: PD >= reject_pd_min
        self.approve_pd_max = self.lr_threshold
        self.reject_pd_min = min(0.85, max(0.50, round(self.lr_threshold * 2.0, 4)))
        self.approve_score_min = self.score_from_pd(self.approve_pd_max)
        self.reject_score_max = self.score_from_pd(self.reject_pd_min)

        if hasattr(self.model, 'intercept_'):
            self.intercept = float(self.model.intercept_[0]) + self.prior_offset
        else:
            self.intercept = self.prior_offset

        if hasattr(self.model, 'feature_names_in_'):
            self.features = list(self.model.feature_names_in_)
        else:
            raise ValueError("Model must have feature_names_in_")

        self.coefs = {f: float(c) for f, c in zip(self.features, self.model.coef_[0])}

    def score_from_pd(self, pd_value: float) -> int:
        """Convert a default probability into the integer credit scorecard scale."""
        clipped = float(np.clip(pd_value, 1e-6, 1.0 - 1e-6))
        logit = np.log(clipped / (1.0 - clipped))
        return int(round(self.offset - self.factor * logit))

    def get_active_base_features(self):
        """Return raw input feature names required by the loaded model."""
        raw_features = []
        for woe_col in self.features:
            base = woe_col.replace('_WOE', '')
            if base.endswith('_binned'):
                raw_features.append(base[:-len('_binned')])
            else:
                raw_features.append(base)
        return raw_features

    def get_valid_categories(self, feature_name: str):
        """Return valid bin labels for a categorical feature from woe_rules.csv."""
        rules = self.woe_rules[self.woe_rules['feature'] == feature_name]
        return [str(b) for b in rules['bin'].tolist()]

    @staticmethod
    def _match_interval(bin_str: str, value: float) -> bool:
        """Evaluate whether a numeric value falls inside a pandas interval or range label."""
        s = str(bin_str).strip()
        if s.startswith('<=') or s.startswith('≤'):
            upper = float(s.lstrip('<=≤').strip())
            return value <= upper
        if s.endswith('+'):
            lower = float(s[:-1].strip())
            return value >= lower
        if (s.startswith('(') or s.startswith('[')) and ',' in s and (s.endswith(')') or s.endswith(']')):
            left_closed = s.startswith('[')
            right_closed = s.endswith(']')
            inner = s[1:-1]
            parts = [p.strip() for p in inner.split(',')]
            if len(parts) == 2:
                lower = float(parts[0])
                upper = float(parts[1])
                left_ok = (value >= lower) if left_closed else (value > lower)
                right_ok = (value <= upper) if right_closed else (value < upper)
                return left_ok and right_ok
        if '-' in s and not s.startswith('-'):
            parts = s.split('-')
            if len(parts) == 2:
                lower, upper = float(parts[0]), float(parts[1])
                return lower <= value <= upper
        return False

    def map_to_woe(self, feature_name, value):
        rules = self.woe_rules[self.woe_rules['feature'] == feature_name]
        if rules.empty:
            return 0.0, "Unknown"

        if feature_name.endswith('_binned'):
            num_val = float(value)
            for _, row in rules.iterrows():
                bin_label = str(row['bin'])
                if self._match_interval(bin_label, num_val):
                    return float(row['WOE']), bin_label
        else:
            match = rules[rules['bin'].astype(str) == str(value)]
            if not match.empty:
                return float(match['WOE'].values[0]), str(value)

        return 0.0, "Unknown"

    def assign_risk_grade(self, prob: float) -> str:
        """Map calibrated probability of default to a 5-tier monotonic risk grade."""
        very_low_max = min(0.05, self.approve_pd_max * 0.5)
        moderate_max = max(self.approve_pd_max + 0.15, 0.30)
        if prob < very_low_max:
            return "Very Low Risk"
        if prob < self.approve_pd_max:
            return "Low Risk"
        if prob < moderate_max:
            return "Moderate Risk"
        if prob < self.reject_pd_min:
            return "High Risk"
        return "Very High Risk"

    def assign_decision(self, prob: float) -> str:
        """Map calibrated probability of default to an underwriting decision anchored to lr_threshold."""
        if prob < self.approve_pd_max:
            return "APPROVE"
        if prob < self.reject_pd_min:
            return "MANUAL REVIEW"
        return "REJECT"

    def explain_customer(self, customer_dict):
        score = self.offset - self.factor * self.intercept
        contributions = []
        log_odds_sum = self.intercept

        for woe_col in self.features:
            base_feature = woe_col.replace('_WOE', '')
            raw_key = base_feature[:-len('_binned')] if base_feature.endswith('_binned') else base_feature
            val = customer_dict.get(raw_key)
            woe_val, bin_val = self.map_to_woe(base_feature, val)

            coef = self.coefs[woe_col]
            log_odds_contrib = coef * woe_val
            log_odds_sum += log_odds_contrib

            contrib = -self.factor * log_odds_contrib
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

        prob = float(1.0 / (1.0 + np.exp(-log_odds_sum)))
        final_score = int(round(score))
        grade = self.assign_risk_grade(prob)
        decision = self.assign_decision(prob)

        sorted_contribs = sorted(contributions, key=lambda x: x['score_contribution'])
        risk_drivers = sorted_contribs[:5]
        positive_drivers = sorted(contributions, key=lambda x: x['score_contribution'], reverse=True)[:5]

        return {
            'Credit Score': final_score,
            'Probability of Default': prob,
            'Risk Grade': grade,
            'Decision': decision,
            'Approve Score Min': self.approve_score_min,
            'Reject Score Max': self.reject_score_max,
            'Risk Drivers': risk_drivers,
            'Positive Drivers': positive_drivers,
            'All Contributions': contributions
        }

    def format_explanation(self, result):
        lines = [
            f"Credit Score: {result['Credit Score']}",
            f"Probability of Default: {result['Probability of Default'] * 100:.1f}%",
            f"Risk Grade: {result['Risk Grade'].upper()}",
            f"Decision: {result['Decision']}\n",
            "Risk Drivers:"
        ]
        for i, driver in enumerate(result['Risk Drivers'], 1):
            if driver['score_contribution'] < 0:
                lines.append(
                    f"{i}. {driver['feature']} (Value: {driver['original_value']}, Bin: {driver['bin']}) "
                    f"-> {driver['score_contribution']:.1f} points"
                )

        lines.append("\nPositive Drivers:")
        for i, driver in enumerate(result['Positive Drivers'], 1):
            if driver['score_contribution'] > 0:
                lines.append(
                    f"{i}. {driver['feature']} (Value: {driver['original_value']}, Bin: {driver['bin']}) "
                    f"-> +{driver['score_contribution']:.1f} points"
                )

        lines.append(
            "\n* Note: These are model-derived risk factors, not causal claims. "
            "Score boundaries and parameters (Base 600, PDO 20) are calibrated via Bayes prior-odds correction."
        )
        return "\n".join(lines)


if __name__ == "__main__":
    explainer = ScorecardExplainer(
        model_path=os.path.join("artifacts", "model_trainer", "model.joblib"),
        woe_rules_path=os.path.join("artifacts", "data_transformation", "woe_rules.csv"),
        threshold_path=os.path.join("artifacts", "model_trainer", "lr_threshold.json"),
    )

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

# Model Calibration & Scorecard Report

## 1. Reliability & Calibration
**Brier Score:** 0.1585

### Reliability Table
| Predicted PD   | Actual Default Rate   |
|:---------------|:----------------------|
| 2.67%          | 0.00%                 |
| 5.49%          | 13.33%                |
| 8.87%          | 13.33%                |
| 13.60%         | 6.67%                 |
| 19.74%         | 20.00%                |
| 26.47%         | 26.67%                |
| 33.32%         | 33.33%                |
| 42.39%         | 53.33%                |
| 57.30%         | 66.67%                |
| 72.97%         | 66.67%                |

## 2. Scorecard Construction
**Illustrative Base Score:** 600
**Illustrative Base Odds:** 50:1
**Points to Double the Odds (PDO):** 20

**Calculated Offset:** 487.12
**Calculated Factor:** 28.85
**Base Points (Offset + Intercept):** 511

### Scorecard Sample (Top 10)
| Feature          | Bin   |        WOE |   Coefficient |   Points |
|:-----------------|:------|-----------:|--------------:|---------:|
| checking_account | A11   | -0.763916  |     -0.777002 |      -17 |
| checking_account | A12   | -0.415165  |     -0.777002 |       -9 |
| checking_account | A13   |  0.133531  |     -0.777002 |        3 |
| checking_account | A14   |  1.18413   |     -0.777002 |       27 |
| credit_history   | A30   | -1.13498   |     -0.588866 |      -19 |
| credit_history   | A31   | -1.05861   |     -0.588866 |      -18 |
| credit_history   | A32   | -0.0828417 |     -0.588866 |       -1 |
| credit_history   | A33   | -0.131678  |     -0.588866 |       -2 |
| credit_history   | A34   |  0.699082  |     -0.588866 |       12 |
| purpose          | A40   | -0.180353  |     -0.829479 |       -4 |

## 3. Score Interpretation Rules
- Score >= 750: Very Low Risk
- 700-749: Low Risk
- 650-699: Moderate Risk
- 600-649: High Risk
- < 600: Very High Risk

## 4. Customer-Level Explanations
### Example 1
```json
{
  "Credit Score": 543,
  "Probability of Default": "12.8%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -10.738799504610732
    },
    {
      "Feature": "age_binned",
      "Value": "18-25",
      "Points": -7.234441153760886
    },
    {
      "Feature": "personal_status_sex",
      "Value": "A92",
      "Points": -4.593337774504535
    },
    {
      "Feature": "savings_account",
      "Value": "A62",
      "Points": -2.1943183080759505
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 26.54776254998758
    },
    {
      "Feature": "purpose",
      "Value": "A41",
      "Points": 13.295795641205403
    },
    {
      "Feature": "duration_binned",
      "Value": "(12.0, 15.0]",
      "Points": 13.076117463162236
    },
    {
      "Feature": "other_installment_plans",
      "Value": "A143",
      "Points": 3.761794771802136
    }
  ]
}
```
### Example 2
```json
{
  "Credit Score": 530,
  "Probability of Default": "18.5%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -10.738799504610732
    },
    {
      "Feature": "property",
      "Value": "A124",
      "Points": -8.549495329250574
    },
    {
      "Feature": "savings_account",
      "Value": "A61",
      "Points": -6.753088327273089
    },
    {
      "Feature": "housing",
      "Value": "A153",
      "Points": -5.599573914138819
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 26.54776254998758
    },
    {
      "Feature": "credit_history",
      "Value": "A34",
      "Points": 11.87815746088598
    },
    {
      "Feature": "purpose",
      "Value": "A43",
      "Points": 9.210789784038289
    },
    {
      "Feature": "other_installment_plans",
      "Value": "A143",
      "Points": 3.761794771802136
    }
  ]
}
```

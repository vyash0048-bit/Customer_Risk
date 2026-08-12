# Model Calibration & Scorecard Report

## 1. Reliability & Calibration
**Brier Score:** 0.1609

### Reliability Table
| Predicted PD   | Actual Default Rate   |
|:---------------|:----------------------|
| 5.64%          | 0.00%                 |
| 9.09%          | 20.00%                |
| 12.71%         | 0.00%                 |
| 17.46%         | 26.67%                |
| 23.55%         | 6.67%                 |
| 29.12%         | 33.33%                |
| 35.09%         | 33.33%                |
| 43.08%         | 46.67%                |
| 54.50%         | 60.00%                |
| 66.20%         | 73.33%                |

## 2. Scorecard Construction
**Illustrative Base Score:** 600
**Illustrative Base Odds:** 50:1
**Points to Double the Odds (PDO):** 20

**Calculated Offset:** 487.12
**Calculated Factor:** 28.85
**Base Points (Offset + Intercept):** 509

### Scorecard Sample (Top 10)
| Feature          | Bin   |        WOE |   Coefficient |   Points |
|:-----------------|:------|-----------:|--------------:|---------:|
| checking_account | A11   | -0.763916  |     -0.703998 |      -16 |
| checking_account | A12   | -0.415165  |     -0.703998 |       -8 |
| checking_account | A13   |  0.133531  |     -0.703998 |        3 |
| checking_account | A14   |  1.18413   |     -0.703998 |       24 |
| credit_history   | A30   | -1.13498   |     -0.501753 |      -16 |
| credit_history   | A31   | -1.05861   |     -0.501753 |      -15 |
| credit_history   | A32   | -0.0828417 |     -0.501753 |       -1 |
| credit_history   | A33   | -0.131678  |     -0.501753 |       -2 |
| credit_history   | A34   |  0.699082  |     -0.501753 |       10 |
| purpose          | A40   | -0.180353  |     -0.534683 |       -3 |

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
  "Credit Score": 537,
  "Probability of Default": "15.0%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -7.589931624552889
    },
    {
      "Feature": "age_binned",
      "Value": "18-25",
      "Points": -4.181150634189743
    },
    {
      "Feature": "personal_status_sex",
      "Value": "A92",
      "Points": -1.890262543687239
    },
    {
      "Feature": "savings_account",
      "Value": "A62",
      "Points": -1.5697047585401507
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 24.053431421679694
    },
    {
      "Feature": "duration_binned",
      "Value": "(12.0, 15.0]",
      "Points": 9.167294406516596
    },
    {
      "Feature": "purpose",
      "Value": "A41",
      "Points": 8.570485326972559
    },
    {
      "Feature": "other_installment_plans",
      "Value": "A143",
      "Points": 2.2909978915345683
    }
  ]
}
```
### Example 2
```json
{
  "Credit Score": 526,
  "Probability of Default": "20.5%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -7.589931624552889
    },
    {
      "Feature": "property",
      "Value": "A124",
      "Points": -7.12787858551759
    },
    {
      "Feature": "savings_account",
      "Value": "A61",
      "Points": -4.830819140117027
    },
    {
      "Feature": "housing",
      "Value": "A153",
      "Points": -4.037005492510004
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 24.053431421679694
    },
    {
      "Feature": "credit_history",
      "Value": "A34",
      "Points": 10.120980965936699
    },
    {
      "Feature": "purpose",
      "Value": "A43",
      "Points": 5.937285802534498
    },
    {
      "Feature": "other_installment_plans",
      "Value": "A143",
      "Points": 2.2909978915345683
    }
  ]
}
```

# Model Calibration & Scorecard Report

## 1. Reliability & Calibration
**Brier Score:** 0.1746

### Reliability Table
| Predicted PD   | Actual Default Rate   |
|:---------------|:----------------------|
| 7.20%          | 0.00%                 |
| 12.73%         | 13.33%                |
| 19.34%         | 6.67%                 |
| 27.19%         | 26.67%                |
| 37.07%         | 0.00%                 |
| 44.91%         | 40.00%                |
| 52.55%         | 33.33%                |
| 62.62%         | 46.67%                |
| 74.12%         | 60.00%                |
| 84.33%         | 73.33%                |

## 2. Scorecard Construction
**Illustrative Base Score:** 600
**Illustrative Base Odds:** 50:1
**Points to Double the Odds (PDO):** 20

**Calculated Offset:** 487.12
**Calculated Factor:** 28.85
**Base Points (Offset + Intercept):** 488

### Scorecard Sample (Top 10)
| Feature          | Bin   |        WOE |   Coefficient |   Points |
|:-----------------|:------|-----------:|--------------:|---------:|
| checking_account | A11   | -0.763916  |     -0.813557 |      -18 |
| checking_account | A12   | -0.415165  |     -0.813557 |      -10 |
| checking_account | A13   |  0.133531  |     -0.813557 |        3 |
| checking_account | A14   |  1.18413   |     -0.813557 |       28 |
| credit_history   | A30   | -1.13498   |     -0.657273 |      -22 |
| credit_history   | A31   | -1.05861   |     -0.657273 |      -20 |
| credit_history   | A32   | -0.0828417 |     -0.657273 |       -2 |
| credit_history   | A33   | -0.131678  |     -0.657273 |       -2 |
| credit_history   | A34   |  0.699082  |     -0.657273 |       13 |
| purpose          | A40   | -0.180353  |     -0.645283 |       -3 |

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
  "Credit Score": 520,
  "Probability of Default": "24.1%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -8.927090203735
    },
    {
      "Feature": "age_binned",
      "Value": "18-25",
      "Points": -4.826632194603725
    },
    {
      "Feature": "personal_status_sex",
      "Value": "A92",
      "Points": -2.6176964321374103
    },
    {
      "Feature": "savings_account",
      "Value": "A62",
      "Points": -2.1346382908526578
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 27.796723267541218
    },
    {
      "Feature": "duration_binned",
      "Value": "(12.0, 15.0]",
      "Points": 11.43637348545366
    },
    {
      "Feature": "purpose",
      "Value": "A41",
      "Points": 10.343295134766835
    },
    {
      "Feature": "other_installment_plans",
      "Value": "A143",
      "Points": 1.8722564541532474
    }
  ]
}
```
### Example 2
```json
{
  "Credit Score": 509,
  "Probability of Default": "32.2%",
  "Risk Grade": "Very High Risk",
  "Risk Drivers": [
    {
      "Feature": "employment",
      "Value": "A72",
      "Points": -8.927090203735
    },
    {
      "Feature": "property",
      "Value": "A124",
      "Points": -8.391266658073537
    },
    {
      "Feature": "savings_account",
      "Value": "A61",
      "Points": -6.569421068881821
    },
    {
      "Feature": "duration_binned",
      "Value": "(15.0, 24.0]",
      "Points": -4.440842667200087
    }
  ],
  "Positive Drivers": [
    {
      "Feature": "checking_account",
      "Value": "A14",
      "Points": 27.796723267541218
    },
    {
      "Feature": "credit_history",
      "Value": "A34",
      "Points": 13.258002509388033
    },
    {
      "Feature": "purpose",
      "Value": "A43",
      "Points": 7.165416777718018
    },
    {
      "Feature": "age_binned",
      "Value": "46-55",
      "Points": 2.4948053735316353
    }
  ]
}
```

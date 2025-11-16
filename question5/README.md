# Question 5: Fitness Data Multiple Regression Analysis

## Problem Description

Predict oxygen consumption rate (ml/kg/min) from fitness measurements in a physical fitness course.

### Variables
- **Y**: Oxygen consumption (ml/kg/min) - **Target variable**
- **X1**: Age (years)
- **X2**: Weight (kg)
- **X3**: Time to run 1.5 miles (minutes)
- **X4**: Resting pulse rate
- **X5**: Pulse rate at begin of run
- **X6**: Pulse rate at end of run

### Dataset
31 observations with 6 predictor variables

---

## Analysis Tasks

### (a) Full Linear Model
Fit the complete model with all 6 predictors:
```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄ + β₅X₅ + β₆X₆ + ε
```
- Estimate all coefficients and σ² using least squares
- Test coefficient significance
- Evaluate overall model fit

### (b) Variable Selection Methods
Apply three methods to select the best subset of predictors:
1. **Forward Selection**: Start with no variables, add one at a time
2. **Backward Elimination**: Start with all variables, remove one at a time
3. **Stepwise Regression**: Combination of forward and backward

### (c) Quadratic Model with Interactions
Fit expanded model including:
- All linear terms (X₁ to X₆)
- All quadratic terms (X₁², X₂², ..., X₆²)
- All interaction terms (X₁X₂, X₁X₃, ..., X₅X₆)

Total: 27 features (6 linear + 6 quadratic + 15 interactions)

### (d) Variable Selection for Quadratic Model
Apply selection methods to the quadratic model and compare with linear model using:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Adjusted R²

---

## Key Results

### Full Linear Model Results
```
Y = 102.24 - 0.22X₁ - 0.07X₂ - 2.68X₃ - 0.00X₄ - 0.37X₅ + 0.30X₆

Statistics:
- R² = 0.848
- Adjusted R² = 0.810
- RMSE = 2.322
- F-statistic = 3712.76 (p < 0.001)

Significant Predictors:
- X1 (Age): β = -0.22, p = 0.037 *
- X3 (Run Time): β = -2.68, p < 0.001 ***
- X5 (Begin Pulse): β = -0.37, p = 0.005 **
- X6 (End Pulse): β = 0.30, p = 0.036 *
```

### Variable Selection Summary
| Method | Selected Variables | R² | Adj R² |
|--------|-------------------|-----|---------|
| Forward Selection | X3 | 0.743 | 0.735 |
| Backward Elimination | X1, X3, X5, X6 | 0.848 | 0.810 |
| Stepwise Regression | X3 | 0.743 | 0.735 |

**Key Finding**: X3 (run time) is the most important predictor, appearing in all methods

### Model Comparison
| Model | Features | R² | Adj R² | AIC | BIC |
|-------|----------|-----|---------|-----|-----|
| Linear (Full) | 6 | 0.848 | 0.810 | 54.30 | 55.79 |
| Quadratic (Full) | 27 | 0.976 | 0.763 | 115.15 | 155.31 |
| Forward (X3 only) | 1 | 0.743 | 0.735 | 66.60 | 69.47 |

**Recommendation**: Linear model with selected variables (X1, X3, X5, X6) provides best balance between fit and complexity

---

## Correlation Insights

**Strongest correlations with Y (Oxygen Consumption)**:
- X3 (Run Time): r = -0.86 *** (strong negative - faster run = higher O₂)
- X5 (Begin Pulse): r = -0.40 (moderate negative)
- X4 (Rest Pulse): r = -0.35 (moderate negative)
- X1 (Age): r = -0.30 (moderate negative)

**Multicollinearity concern**:
- X5 and X6 (pulse rates): r = 0.93 (very high correlation)
- This may cause instability in coefficient estimates

---

## Interpretation

1. **Run Time (X3)** is the dominant predictor - individuals who run faster consume more oxygen (better fitness)

2. **Age (X1)** has negative effect - older individuals tend to have lower oxygen consumption

3. **Pulse Rates (X5, X6)** show complex relationship - high correlation between them suggests they measure similar physiological response

4. **Weight (X2)** and **Resting Pulse (X4)** are not significant in the full model

5. **Quadratic model** overfits (high R² but low adjusted R² and high BIC)

---

## Files Generated

- `fitness_analysis.py` - Complete analysis script
- `correlation_matrix.png` - Heatmap of variable correlations
- `diagnostic_plots.png` - Model diagnostic plots (residuals, Q-Q plot, etc.)
- `README.md` - This summary document

---

## Running the Analysis

```bash
cd question5
uv run fitness_analysis.py
```

Or using Python directly:
```bash
cd question5
python fitness_analysis.py
```

---

## Statistical Methods Used

- **Multiple Linear Regression**: Ordinary Least Squares (OLS)
- **Forward Selection**: p-value threshold = 0.05
- **Backward Elimination**: p-value threshold = 0.10
- **Stepwise Regression**: Combined forward/backward
- **Model Selection Criteria**: AIC, BIC, Adjusted R²
- **Polynomial Features**: degree=2 with interactions
- **Hypothesis Testing**: t-tests, F-test
- **Diagnostics**: Residual plots, Q-Q plot, normality tests

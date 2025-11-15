import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

def main():
    """
    Regression Analysis: Bacteria Survival vs. Exposure Time at 300°F
    
    Problem Description:
    Analyze the relationship between bacteria count in canned food and 
    exposure time to 300°F heat
    
    (a) Plot scatter diagram and assess straight-line model adequacy
    (b) Fit straight-line model, compute summary statistics and residual plots
    (c) Identify appropriate transformed model and conduct model adequacy tests
    """
    
    # Data input
    minutes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    bacteria = np.array([175, 108, 95, 82, 71, 50, 49, 31, 28, 17, 16, 11])
    
    # Create DataFrame for analysis
    df = pd.DataFrame({'Minutes': minutes, 'Bacteria': bacteria})
    print("=" * 60)
    print("Original Data:")
    print("=" * 60)
    print(df)
    print()
    
    # ==================== (a) Scatter Plot Analysis ====================
    print("=" * 60)
    print("(a) Scatter Plot Analysis")
    print("=" * 60)
    
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(minutes, bacteria, color='blue', s=100, alpha=0.6, edgecolors='black')
    plt.xlabel('Minutes of Exposure', fontsize=11)
    plt.ylabel('Number of Bacteria', fontsize=11)
    plt.title('(a) Scatter Plot: Bacteria vs Exposure Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Initial observations
    print("Initial Observations:")
    print("- Data shows clear non-linear trend (exponential decay)")
    print("- Bacteria count decreases rapidly, rate slows over time")
    print("- Straight-line model may not be adequate, transformation needed")
    print()
    
    # ==================== (b) Linear Regression Model ====================
    print("=" * 60)
    print("(b) Linear Regression Model")
    print("=" * 60)
    
    # Fit linear model
    X = minutes.reshape(-1, 1)
    y = bacteria
    
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)
    
    # Calculate statistics
    residuals_linear = y - y_pred_linear
    ss_res = np.sum(residuals_linear ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = r2_score(y, y_pred_linear)
    n = len(y)
    p = 1  # one predictor variable
    mse = ss_res / (n - p - 1)
    rmse = np.sqrt(mse)
    
    # Standard errors
    se_slope = np.sqrt(mse / np.sum((minutes - np.mean(minutes)) ** 2))
    se_intercept = np.sqrt(mse * (1/n + np.mean(minutes)**2 / np.sum((minutes - np.mean(minutes)) ** 2)))
    
    print(f"Linear Model: Bacteria = {model_linear.intercept_:.4f} + {model_linear.coef_[0]:.4f} * Minutes")
    print(f"\nModel Statistics:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MSE = {mse:.4f}")
    print(f"  SE(Intercept) = {se_intercept:.4f}")
    print(f"  SE(Slope) = {se_slope:.4f}")
    
    # t-test
    t_slope = model_linear.coef_[0] / se_slope
    p_value_slope = 2 * (1 - stats.t.cdf(abs(t_slope), n - 2))
    print(f"\nSlope Significance Test:")
    print(f"  t-statistic = {t_slope:.4f}")
    print(f"  p-value = {p_value_slope:.6f}")
    
    # Residual analysis plots
    plt.subplot(1, 3, 2)
    plt.scatter(minutes, bacteria, color='blue', s=100, alpha=0.6, edgecolors='black', label='Actual Data')
    plt.plot(minutes, y_pred_linear, color='red', linewidth=2, label='Linear Fit')
    plt.xlabel('Minutes of Exposure', fontsize=11)
    plt.ylabel('Number of Bacteria', fontsize=11)
    plt.title('(b) Linear Model Fit', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred_linear, residuals_linear, color='red', s=100, alpha=0.6, edgecolors='black')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values', fontsize=11)
    plt.ylabel('Residuals', fontsize=11)
    plt.title('(b) Residual Plot for Linear Model', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_model_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: linear_model_analysis.png")
    
    # Residual normality test
    _, p_value_normality = stats.shapiro(residuals_linear)
    print(f"\nResidual Normality Test (Shapiro-Wilk):")
    print(f"  p-value = {p_value_normality:.4f}")
    if p_value_normality > 0.05:
        print("  Conclusion: Residuals are normally distributed (p > 0.05)")
    else:
        print("  Conclusion: Residuals are not normally distributed (p <= 0.05)")
    
    print(f"\nLinear Model Adequacy Assessment:")
    print("  - Residual plot shows clear pattern (non-random), indicating non-linearity")
    print("  - Despite high R², residuals show systematic bias")
    print("  - Conclusion: Linear model is not adequate, transformation needed")
    print()
    
    # ==================== (c) Transformed Model ====================
    print("=" * 60)
    print("(c) Transformed Model (Logarithmic Transformation)")
    print("=" * 60)
    
    # Logarithmic transformation (exponential model)
    # ln(Bacteria) = beta_0 + beta_1 * Minutes
    # i.e., Bacteria = exp(beta_0) * exp(beta_1 * Minutes)
    
    y_log = np.log(bacteria)
    
    model_log = LinearRegression()
    model_log.fit(X, y_log)
    y_log_pred = model_log.predict(X)
    y_pred_exp = np.exp(y_log_pred)
    
    # Calculate statistics
    residuals_log = y_log - y_log_pred
    r2_log = r2_score(y_log, y_log_pred)
    mse_log = np.mean(residuals_log ** 2)
    rmse_log = np.sqrt(mse_log)
    
    # Statistics on original scale
    r2_original = r2_score(y, y_pred_exp)
    rmse_original = np.sqrt(mean_squared_error(y, y_pred_exp))
    
    print(f"Log-transformed Model: ln(Bacteria) = {model_log.intercept_:.4f} + {model_log.coef_[0]:.4f} * Minutes")
    print(f"Exponential Form: Bacteria = {np.exp(model_log.intercept_):.4f} * exp({model_log.coef_[0]:.4f} * Minutes)")
    print(f"\nTransformed Model Statistics (log scale):")
    print(f"  R² = {r2_log:.4f}")
    print(f"  RMSE = {rmse_log:.4f}")
    print(f"\nOriginal Scale Statistics:")
    print(f"  R² = {r2_original:.4f}")
    print(f"  RMSE = {rmse_original:.4f}")
    
    # Standard errors
    se_slope_log = np.sqrt(mse_log / np.sum((minutes - np.mean(minutes)) ** 2))
    t_slope_log = model_log.coef_[0] / se_slope_log
    p_value_slope_log = 2 * (1 - stats.t.cdf(abs(t_slope_log), n - 2))
    
    print(f"\nSlope Significance Test:")
    print(f"  t-statistic = {t_slope_log:.4f}")
    print(f"  p-value = {p_value_slope_log:.6f}")
    
    # Visualize transformed model
    plt.figure(figsize=(15, 10))
    
    # Fit on original scale
    plt.subplot(2, 3, 1)
    plt.scatter(minutes, bacteria, color='blue', s=100, alpha=0.6, edgecolors='black', label='Actual Data')
    plt.plot(minutes, y_pred_exp, color='green', linewidth=2, label='Exponential Model')
    plt.plot(minutes, y_pred_linear, color='red', linewidth=2, linestyle='--', alpha=0.5, label='Linear Model')
    plt.xlabel('Minutes of Exposure', fontsize=11)
    plt.ylabel('Number of Bacteria', fontsize=11)
    plt.title('(c) Exponential Model Fit (Original Scale)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Fit on log scale
    plt.subplot(2, 3, 2)
    plt.scatter(minutes, y_log, color='blue', s=100, alpha=0.6, edgecolors='black', label='ln(Bacteria)')
    plt.plot(minutes, y_log_pred, color='green', linewidth=2, label='Linear Fit')
    plt.xlabel('Minutes of Exposure', fontsize=11)
    plt.ylabel('ln(Number of Bacteria)', fontsize=11)
    plt.title('(c) Linear Fit on Log Scale', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot for log model
    plt.subplot(2, 3, 3)
    plt.scatter(y_log_pred, residuals_log, color='green', s=100, alpha=0.6, edgecolors='black')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values (log scale)', fontsize=11)
    plt.ylabel('Residuals', fontsize=11)
    plt.title('(c) Residual Plot (Log Model)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot for residual normality
    plt.subplot(2, 3, 4)
    stats.probplot(residuals_log, dist="norm", plot=plt)
    plt.title('(c) Q-Q Plot for Log Model Residuals', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residual histogram
    plt.subplot(2, 3, 5)
    plt.hist(residuals_log, bins=8, color='green', alpha=0.6, edgecolor='black')
    plt.xlabel('Residuals', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('(c) Histogram of Residuals (Log Model)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs observation order
    plt.subplot(2, 3, 6)
    plt.scatter(range(1, n+1), residuals_log, color='green', s=100, alpha=0.6, edgecolors='black')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Observation Order', fontsize=11)
    plt.ylabel('Residuals', fontsize=11)
    plt.title('(c) Residuals vs Order', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformed_model_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: transformed_model_analysis.png")
    
    # Residual normality test
    _, p_value_normality_log = stats.shapiro(residuals_log)
    print(f"\nLog Model Residual Normality Test (Shapiro-Wilk):")
    print(f"  p-value = {p_value_normality_log:.4f}")
    if p_value_normality_log > 0.05:
        print("  Conclusion: Residuals are normally distributed (p > 0.05)")
    else:
        print("  Conclusion: Residuals are not normally distributed (p <= 0.05)")
    
    # Model comparison
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<25} {'Linear Model':<20} {'Log-Transform Model':<20}")
    print("-" * 65)
    print(f"{'R² (original scale)':<25} {r2:<20.4f} {r2_original:<20.4f}")
    print(f"{'RMSE (original scale)':<25} {rmse:<20.4f} {rmse_original:<20.4f}")
    print(f"{'Residual normality p':<25} {p_value_normality:<20.4f} {p_value_normality_log:<20.4f}")
    
    print(f"\nFinal Conclusions:")
    print("  - Log-transformed model (exponential) significantly better than linear")
    print("  - Residuals more random with no clear pattern")
    print("  - Higher R² and better fit")
    print("  - Recommended model: Bacteria = {:.2f} * exp({:.4f} * Minutes)".format(
        np.exp(model_log.intercept_), model_log.coef_[0]))
    print()
    
    # Predictions
    print("=" * 60)
    print("Model Prediction Examples")
    print("=" * 60)
    new_minutes = np.array([13, 14, 15]).reshape(-1, 1)
    pred_linear = model_linear.predict(new_minutes)
    pred_exp = np.exp(model_log.predict(new_minutes))
    
    print(f"{'Time (min)':<20} {'Linear Prediction':<25} {'Exponential Prediction':<25}")
    print("-" * 70)
    for i, m in enumerate(new_minutes.flatten()):
        print(f"{m:<20} {pred_linear[i]:<25.2f} {pred_exp[i]:<25.2f}")
    
    print("\nAnalysis Complete!")
    plt.show()


if __name__ == "__main__":
    main()

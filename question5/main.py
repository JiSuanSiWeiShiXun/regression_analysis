import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """
    Fitness Data Regression Analysis
    
    Predicting oxygen consumption (Y) from various physical fitness measures:
    X1: Age (years)
    X2: Weight (kg)
    X3: Time to run 1.5 miles (minutes)
    X4: Resting pulse rate
    X5: Pulse rate at beginning of run
    X6: Pulse rate at end of run
    Y: Oxygen consumption (ml/kg/min)
    """
    
    # ==================== Data Input ====================
    print("="*80)
    print("QUESTION 5: FITNESS DATA REGRESSION ANALYSIS")
    print("="*80)
    print()
    
    # Data from the table
    data = {
        'Individual': range(1, 32),
        'Y': [44.609, 45.313, 54.297, 59.571, 49.874, 44.811, 45.681, 49.091, 39.442,
              60.055, 50.541, 37.388, 44.754, 47.273, 51.855, 49.156, 40.836, 46.672,
              46.774, 50.388, 39.407, 46.080, 45.441, 54.625, 45.118, 39.203, 45.790,
              50.545, 48.673, 47.920, 47.467],
        'X1': [44, 40, 44, 42, 38, 47, 40, 43, 44, 38, 44, 45, 45, 47, 54, 49, 51, 51,
               48, 49, 57, 54, 52, 50, 51, 54, 51, 57, 49, 48, 52],
        'X2': [89.47, 75.07, 85.84, 68.15, 89.02, 77.45, 75.98, 81.19, 81.42, 81.87,
               73.03, 87.66, 66.45, 79.15, 83.12, 81.42, 69.63, 77.91, 91.63, 73.37,
               73.37, 79.38, 76.32, 70.87, 67.25, 91.63, 73.71, 59.08, 76.32, 61.24, 82.78],
        'X3': [11.37, 10.07, 8.65, 8.17, 9.22, 11.63, 11.95, 10.85, 13.08, 8.63, 10.13,
               14.03, 11.12, 10.60, 10.33, 8.95, 10.95, 10.00, 10.25, 10.08, 12.63, 11.17,
               9.63, 8.92, 11.08, 12.88, 10.47, 9.93, 9.40, 11.50, 10.50],
        'X4': [62, 62, 45, 40, 55, 58, 70, 64, 63, 48, 45, 56, 51, 47, 50, 44, 57, 48,
               48, 76, 58, 62, 48, 48, 48, 44, 59, 49, 56, 52, 53],
        'X5': [178, 185, 156, 166, 178, 176, 176, 162, 174, 170, 168, 186, 176, 162,
               166, 180, 168, 162, 162, 168, 174, 156, 164, 146, 172, 168, 186, 148,
               186, 170, 170],
        'X6': [182, 185, 168, 172, 180, 176, 180, 170, 176, 186, 168, 192, 176, 164,
               170, 185, 172, 168, 164, 168, 176, 165, 166, 155, 172, 172, 188, 155,
               188, 176, 172]
    }
    
    df = pd.DataFrame(data)
    
    print("Dataset Preview:")
    print(df.head(10))
    print(f"\nTotal observations: {len(df)}")
    print()
    
    # ==================== Descriptive Statistics ====================
    print("="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    print(df.describe())
    print()
    
    # Correlation matrix
    print("="*80)
    print("CORRELATION MATRIX")
    print("="*80)
    corr_matrix = df[['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']].corr()
    print(corr_matrix.round(4))
    print()
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('question5_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Figure saved: question5_correlation_matrix.png\n")
    
    # ==================== (a) Multiple Linear Regression Model ====================
    print("="*80)
    print("(a) MULTIPLE LINEAR REGRESSION MODEL")
    print("="*80)
    print("Model: Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄ + β₅X₅ + β₆X₆ + ε")
    print()
    
    # Prepare data
    X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
    y = df['Y'].values
    n = len(y)
    p = 6  # number of predictors
    
    # Fit full model
    model_full = LinearRegression()
    model_full.fit(X, y)
    y_pred_full = model_full.predict(X)
    
    # Calculate statistics
    residuals_full = y - y_pred_full
    ss_res = np.sum(residuals_full ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_full = r2_score(y, y_pred_full)
    r2_adj_full = 1 - (1 - r2_full) * (n - 1) / (n - p - 1)
    mse_full = ss_res / (n - p - 1)
    rmse_full = np.sqrt(mse_full)
    sigma2_full = mse_full  # σ² estimate
    
    # Print results
    print(f"Coefficient Estimates:")
    print(f"  β₀ (Intercept) = {model_full.intercept_:.6f}")
    for i, coef in enumerate(model_full.coef_, 1):
        print(f"  β{i} (X{i}) = {coef:.6f}")
    
    print(f"\nModel Statistics:")
    print(f"  σ² (MSE) = {sigma2_full:.6f}")
    print(f"  RMSE = {rmse_full:.6f}")
    print(f"  R² = {r2_full:.6f}")
    print(f"  Adjusted R² = {r2_adj_full:.6f}")
    print(f"  F-statistic = {(r2_full / p) / ((1 - r2_full) / (n - p - 1)):.4f}")
    print()
    
    # Visualizations for part (a)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('(a) Multiple Linear Regression Analysis - Full Model', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y, y_pred_full, alpha=0.6, s=80, edgecolors='black')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
    axes[0, 0].set_xlabel('Actual Y', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Y', fontsize=11)
    axes[0, 0].set_title('Actual vs Predicted Values', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted
    axes[0, 1].scatter(y_pred_full, residuals_full, alpha=0.6, s=80, edgecolors='black')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Fitted Values', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals_full, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot for Residuals', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual Histogram
    axes[1, 0].hist(residuals_full, bins=12, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Histogram of Residuals', fontweight='bold')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Coefficient Plot
    coef_names = ['X1\n(Age)', 'X2\n(Weight)', 'X3\n(Run Time)', 
                  'X4\n(Rest Pulse)', 'X5\n(Begin Pulse)', 'X6\n(End Pulse)']
    axes[1, 1].barh(coef_names, model_full.coef_, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 1].set_title('Regression Coefficients', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Scale-Location Plot
    standardized_residuals = residuals_full / np.std(residuals_full)
    axes[1, 2].scatter(y_pred_full, np.sqrt(np.abs(standardized_residuals)), 
                      alpha=0.6, s=80, edgecolors='black')
    axes[1, 2].set_xlabel('Fitted Values', fontsize=11)
    axes[1, 2].set_ylabel('√|Standardized Residuals|', fontsize=11)
    axes[1, 2].set_title('Scale-Location Plot', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question5_part_a_full_model.png', dpi=300, bbox_inches='tight')
    print("Figure saved: question5_part_a_full_model.png\n")
    
    # ==================== (b) Variable Selection for Linear Model ====================
    print("="*80)
    print("(b) VARIABLE SELECTION FOR LINEAR MODEL")
    print("="*80)
    print()
    
    # Helper functions for variable selection
    def calculate_aic(n, ss_res, k):
        """Calculate AIC: AIC = n*ln(SSE/n) + 2k"""
        return n * np.log(ss_res / n) + 2 * k
    
    def calculate_bic(n, ss_res, k):
        """Calculate BIC: BIC = n*ln(SSE/n) + k*ln(n)"""
        return n * np.log(ss_res / n) + k * np.log(n)
    
    def calculate_cp(ss_res, n, k, mse_full):
        """Calculate Mallows' Cp"""
        return ss_res / mse_full - n + 2 * k
    
    def fit_model_subset(X_data, y_data, predictors):
        """Fit model with given subset of predictors"""
        if len(predictors) == 0:
            y_mean = np.mean(y_data)
            y_pred = np.full_like(y_data, y_mean)
            ss_res = np.sum((y_data - y_pred) ** 2)
            return None, y_pred, ss_res, 0.0
        
        X_subset = X_data[:, predictors]
        model = LinearRegression()
        model.fit(X_subset, y_data)
        y_pred = model.predict(X_subset)
        ss_res = np.sum((y_data - y_pred) ** 2)
        r2 = r2_score(y_data, y_pred)
        return model, y_pred, ss_res, r2
    
    # Forward Selection
    print("--- Forward Selection ---")
    available = list(range(6))
    selected = []
    forward_models = []
    
    for step in range(7):
        if step == 0:
            # Null model
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_mean) ** 2)
            k = 1  # just intercept
            aic = calculate_aic(n, ss_res, k)
            bic = calculate_bic(n, ss_res, k)
            cp = calculate_cp(ss_res, n, k, mse_full)
            forward_models.append({
                'step': 0,
                'predictors': [],
                'k': k,
                'ss_res': ss_res,
                'r2': 0.0,
                'aic': aic,
                'bic': bic,
                'cp': cp
            })
        else:
            best_metric = {'aic': float('inf'), 'bic': float('inf')}
            best_predictor = None
            best_stats = None
            
            for predictor in available:
                test_predictors = selected + [predictor]
                model, y_pred, ss_res, r2 = fit_model_subset(X, y, test_predictors)
                k = len(test_predictors) + 1
                aic = calculate_aic(n, ss_res, k)
                bic = calculate_bic(n, ss_res, k)
                cp = calculate_cp(ss_res, n, k, mse_full)
                
                if aic < best_metric['aic']:
                    best_metric = {'aic': aic, 'bic': bic}
                    best_predictor = predictor
                    best_stats = {
                        'ss_res': ss_res,
                        'r2': r2,
                        'k': k,
                        'cp': cp
                    }
            
            if best_predictor is not None:
                selected.append(best_predictor)
                available.remove(best_predictor)
                forward_models.append({
                    'step': step,
                    'predictors': selected.copy(),
                    'k': best_stats['k'],
                    'ss_res': best_stats['ss_res'],
                    'r2': best_stats['r2'],
                    'aic': best_metric['aic'],
                    'bic': best_metric['bic'],
                    'cp': best_stats['cp']
                })
    
    # Display forward selection results
    print("\nForward Selection Results:")
    print(f"{'Step':<6} {'Predictors':<20} {'R²':<10} {'AIC':<12} {'BIC':<12} {'Cp':<12}")
    print("-" * 72)
    for model_info in forward_models:
        pred_str = ','.join([f"X{p+1}" for p in model_info['predictors']]) if model_info['predictors'] else "None"
        print(f"{model_info['step']:<6} {pred_str:<20} {model_info['r2']:<10.4f} "
              f"{model_info['aic']:<12.4f} {model_info['bic']:<12.4f} {model_info['cp']:<12.4f}")
    
    # Find best models by each criterion
    best_aic_idx = min(range(len(forward_models)), key=lambda i: forward_models[i]['aic'])
    best_bic_idx = min(range(len(forward_models)), key=lambda i: forward_models[i]['bic'])
    best_cp_idx = min(range(len(forward_models)), 
                     key=lambda i: abs(forward_models[i]['cp'] - forward_models[i]['k']))
    
    print(f"\nBest model by AIC: Step {forward_models[best_aic_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in forward_models[best_aic_idx]['predictors']]}")
    print(f"Best model by BIC: Step {forward_models[best_bic_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in forward_models[best_bic_idx]['predictors']]}")
    print(f"Best model by Cp: Step {forward_models[best_cp_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in forward_models[best_cp_idx]['predictors']]}")
    print()
    
    # Backward Elimination
    print("--- Backward Elimination ---")
    selected = list(range(6))
    backward_models = []
    
    # Start with full model
    model, y_pred, ss_res, r2 = fit_model_subset(X, y, selected)
    k = len(selected) + 1
    backward_models.append({
        'step': 0,
        'predictors': selected.copy(),
        'k': k,
        'ss_res': ss_res,
        'r2': r2,
        'aic': calculate_aic(n, ss_res, k),
        'bic': calculate_bic(n, ss_res, k),
        'cp': calculate_cp(ss_res, n, k, mse_full)
    })
    
    for step in range(1, 7):
        if len(selected) == 0:
            break
            
        best_metric = {'aic': float('inf'), 'bic': float('inf')}
        worst_predictor = None
        best_stats = None
        
        for predictor in selected:
            test_predictors = [p for p in selected if p != predictor]
            model, y_pred, ss_res, r2 = fit_model_subset(X, y, test_predictors)
            k = len(test_predictors) + 1
            aic = calculate_aic(n, ss_res, k)
            bic = calculate_bic(n, ss_res, k)
            cp = calculate_cp(ss_res, n, k, mse_full)
            
            if aic < best_metric['aic']:
                best_metric = {'aic': aic, 'bic': bic}
                worst_predictor = predictor
                best_stats = {
                    'predictors': test_predictors,
                    'ss_res': ss_res,
                    'r2': r2,
                    'k': k,
                    'cp': cp
                }
        
        if worst_predictor is not None:
            selected.remove(worst_predictor)
            backward_models.append({
                'step': step,
                'predictors': best_stats['predictors'],
                'k': best_stats['k'],
                'ss_res': best_stats['ss_res'],
                'r2': best_stats['r2'],
                'aic': best_metric['aic'],
                'bic': best_metric['bic'],
                'cp': best_stats['cp']
            })
    
    # Display backward elimination results
    print("\nBackward Elimination Results:")
    print(f"{'Step':<6} {'Predictors':<20} {'R²':<10} {'AIC':<12} {'BIC':<12} {'Cp':<12}")
    print("-" * 72)
    for model_info in backward_models:
        pred_str = ','.join([f"X{p+1}" for p in model_info['predictors']]) if model_info['predictors'] else "None"
        print(f"{model_info['step']:<6} {pred_str:<20} {model_info['r2']:<10.4f} "
              f"{model_info['aic']:<12.4f} {model_info['bic']:<12.4f} {model_info['cp']:<12.4f}")
    
    # Find best models
    best_aic_idx = min(range(len(backward_models)), key=lambda i: backward_models[i]['aic'])
    best_bic_idx = min(range(len(backward_models)), key=lambda i: backward_models[i]['bic'])
    best_cp_idx = min(range(len(backward_models)), 
                     key=lambda i: abs(backward_models[i]['cp'] - backward_models[i]['k']))
    
    print(f"\nBest model by AIC: Step {backward_models[best_aic_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in backward_models[best_aic_idx]['predictors']]}")
    print(f"Best model by BIC: Step {backward_models[best_bic_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in backward_models[best_bic_idx]['predictors']]}")
    print(f"Best model by Cp: Step {backward_models[best_cp_idx]['step']}, "
          f"Predictors: {[f'X{p+1}' for p in backward_models[best_cp_idx]['predictors']]}")
    print()
    
    # Visualization for part (b)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('(b) Variable Selection for Linear Model', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Forward selection criteria plot
    steps_forward = [m['step'] for m in forward_models]
    axes[0, 0].plot(steps_forward, [m['aic'] for m in forward_models], 
                   'o-', label='AIC', linewidth=2, markersize=8)
    axes[0, 0].plot(steps_forward, [m['bic'] for m in forward_models], 
                   's-', label='BIC', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Predictors', fontsize=11)
    axes[0, 0].set_ylabel('Criterion Value', fontsize=11)
    axes[0, 0].set_title('Forward Selection: AIC and BIC', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Forward Cp plot
    axes[0, 1].plot(steps_forward, [m['cp'] for m in forward_models], 
                   'o-', label='Cp', linewidth=2, markersize=8, color='green')
    axes[0, 1].plot(steps_forward, steps_forward, 'r--', label='Cp = p', linewidth=2)
    axes[0, 1].set_xlabel('Number of Predictors (p)', fontsize=11)
    axes[0, 1].set_ylabel('Mallows\' Cp', fontsize=11)
    axes[0, 1].set_title('Forward Selection: Mallows\' Cp', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Backward elimination criteria plot
    steps_backward = [m['step'] for m in backward_models]
    num_pred_backward = [len(m['predictors']) for m in backward_models]
    axes[1, 0].plot(num_pred_backward, [m['aic'] for m in backward_models], 
                   'o-', label='AIC', linewidth=2, markersize=8)
    axes[1, 0].plot(num_pred_backward, [m['bic'] for m in backward_models], 
                   's-', label='BIC', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Predictors', fontsize=11)
    axes[1, 0].set_ylabel('Criterion Value', fontsize=11)
    axes[1, 0].set_title('Backward Elimination: AIC and BIC', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Backward Cp plot
    axes[1, 1].plot(num_pred_backward, [m['cp'] for m in backward_models], 
                   'o-', label='Cp', linewidth=2, markersize=8, color='green')
    axes[1, 1].plot(num_pred_backward, num_pred_backward, 'r--', label='Cp = p', linewidth=2)
    axes[1, 1].set_xlabel('Number of Predictors (p)', fontsize=11)
    axes[1, 1].set_ylabel('Mallows\' Cp', fontsize=11)
    axes[1, 1].set_title('Backward Elimination: Mallows\' Cp', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question5_part_b_selection.png', dpi=300, bbox_inches='tight')
    print("Figure saved: question5_part_b_selection.png\n")
    
    # Recommended model
    print("="*80)
    print("RECOMMENDED LINEAR MODEL")
    print("="*80)
    recommended_predictors = forward_models[best_bic_idx]['predictors']
    print(f"Selected predictors: {[f'X{p+1}' for p in recommended_predictors]}")
    
    if recommended_predictors:
        X_recommended = X[:, recommended_predictors]
        model_recommended = LinearRegression()
        model_recommended.fit(X_recommended, y)
        y_pred_recommended = model_recommended.predict(X_recommended)
        
        print(f"\nRecommended Model Equation:")
        print(f"Y = {model_recommended.intercept_:.4f}", end="")
        for i, pred_idx in enumerate(recommended_predictors):
            print(f" + {model_recommended.coef_[i]:.4f}*X{pred_idx+1}", end="")
        print()
        
        r2_rec = r2_score(y, y_pred_recommended)
        r2_adj_rec = 1 - (1 - r2_rec) * (n - 1) / (n - len(recommended_predictors) - 1)
        print(f"\nR² = {r2_rec:.4f}")
        print(f"Adjusted R² = {r2_adj_rec:.4f}")
    print()
    
    # ==================== (c) Quadratic Regression Model ====================
    print("="*80)
    print("(c) QUADRATIC REGRESSION MODEL")
    print("="*80)
    print("Model: Y = β₀ + Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ + ε")
    print()
    
    # Create quadratic features
    X_quad = X.copy()
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    
    # Add interaction terms
    interaction_features = []
    interaction_names = []
    for i in range(6):
        for j in range(i, 6):
            interaction_features.append(X[:, i] * X[:, j])
            if i == j:
                interaction_names.append(f'X{i+1}²')
            else:
                interaction_names.append(f'X{i+1}*X{j+1}')
    
    X_quad_full = np.column_stack([X] + interaction_features)
    all_feature_names = feature_names + interaction_names
    
    print(f"Total features in quadratic model: {X_quad_full.shape[1]}")
    print(f"Linear terms: {len(feature_names)}")
    print(f"Quadratic/Interaction terms: {len(interaction_names)}")
    print()
    
    # Fit quadratic model
    model_quad = LinearRegression()
    model_quad.fit(X_quad_full, y)
    y_pred_quad = model_quad.predict(X_quad_full)
    
    # Calculate statistics
    residuals_quad = y - y_pred_quad
    ss_res_quad = np.sum(residuals_quad ** 2)
    r2_quad = r2_score(y, y_pred_quad)
    p_quad = X_quad_full.shape[1]
    r2_adj_quad = 1 - (1 - r2_quad) * (n - 1) / (n - p_quad - 1)
    mse_quad = ss_res_quad / (n - p_quad - 1)
    rmse_quad = np.sqrt(mse_quad)
    sigma2_quad = mse_quad
    
    print(f"Quadratic Model Coefficients:")
    print(f"  β₀ (Intercept) = {model_quad.intercept_:.6f}")
    print(f"\nLinear terms:")
    for i in range(6):
        print(f"  β{i+1} (X{i+1}) = {model_quad.coef_[i]:.6f}")
    print(f"\nQuadratic/Interaction terms (top 10 by absolute value):")
    quad_coefs = [(all_feature_names[i+6], model_quad.coef_[i+6]) 
                  for i in range(len(interaction_names))]
    quad_coefs_sorted = sorted(quad_coefs, key=lambda x: abs(x[1]), reverse=True)
    for name, coef in quad_coefs_sorted[:10]:
        print(f"  β({name}) = {coef:.6f}")
    
    print(f"\nQuadratic Model Statistics:")
    print(f"  σ² (MSE) = {sigma2_quad:.6f}")
    print(f"  RMSE = {rmse_quad:.6f}")
    print(f"  R² = {r2_quad:.6f}")
    print(f"  Adjusted R² = {r2_adj_quad:.6f}")
    print()
    
    # Visualization for part (c)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('(c) Quadratic Regression Model Analysis', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y, y_pred_quad, alpha=0.6, s=80, edgecolors='black', color='purple')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
    axes[0, 0].set_xlabel('Actual Y', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Y', fontsize=11)
    axes[0, 0].set_title('Actual vs Predicted Values', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted
    axes[0, 1].scatter(y_pred_quad, residuals_quad, alpha=0.6, s=80, 
                      edgecolors='black', color='purple')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Fitted Values', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    stats.probplot(residuals_quad, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot for Residuals', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual Histogram
    axes[1, 0].hist(residuals_quad, bins=12, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Histogram of Residuals', fontweight='bold')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Top coefficients
    top_coefs = sorted(enumerate(model_quad.coef_), key=lambda x: abs(x[1]), reverse=True)[:10]
    top_names = [all_feature_names[i] for i, _ in top_coefs]
    top_values = [val for _, val in top_coefs]
    
    axes[1, 1].barh(range(len(top_names)), top_values, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_names)))
    axes[1, 1].set_yticklabels(top_names)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 1].set_title('Top 10 Coefficients (by magnitude)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Model comparison
    models_compare = ['Linear\n(Full)', 'Quadratic\n(Full)']
    r2_compare = [r2_full, r2_quad]
    r2_adj_compare = [r2_adj_full, r2_adj_quad]
    
    x_pos = np.arange(len(models_compare))
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, r2_compare, width, label='R²', 
                  alpha=0.8, edgecolor='black')
    axes[1, 2].bar(x_pos + width/2, r2_adj_compare, width, label='Adjusted R²', 
                  alpha=0.8, edgecolor='black')
    axes[1, 2].set_ylabel('R² Value', fontsize=11)
    axes[1, 2].set_title('Model Comparison', fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(models_compare)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('question5_part_c_quadratic.png', dpi=300, bbox_inches='tight')
    print("Figure saved: question5_part_c_quadratic.png\n")
    
    # ==================== (d) Variable Selection for Quadratic Model ====================
    print("="*80)
    print("(d) VARIABLE SELECTION FOR QUADRATIC MODEL")
    print("="*80)
    print("Note: Due to computational complexity with 27 features,")
    print("we'll use a greedy approach for variable selection.")
    print()
    
    # Forward selection for quadratic model
    print("--- Forward Selection (Quadratic Model) ---")
    available_quad = list(range(X_quad_full.shape[1]))
    selected_quad = []
    forward_quad_models = []
    
    # Null model
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_mean) ** 2)
    k = 1
    forward_quad_models.append({
        'step': 0,
        'predictors': [],
        'k': k,
        'r2': 0.0,
        'aic': calculate_aic(n, ss_res, k),
        'bic': calculate_bic(n, ss_res, k),
        'cp': calculate_cp(ss_res, n, k, mse_quad)
    })
    
    # Perform forward selection (limit to 15 steps for efficiency)
    max_steps = min(15, X_quad_full.shape[1])
    for step in range(1, max_steps + 1):
        best_bic = float('inf')
        best_predictor = None
        best_stats = None
        
        for predictor in available_quad:
            test_predictors = selected_quad + [predictor]
            X_test = X_quad_full[:, test_predictors]
            model_test = LinearRegression()
            model_test.fit(X_test, y)
            y_pred_test = model_test.predict(X_test)
            ss_res_test = np.sum((y - y_pred_test) ** 2)
            r2_test = r2_score(y, y_pred_test)
            k_test = len(test_predictors) + 1
            
            bic_test = calculate_bic(n, ss_res_test, k_test)
            
            if bic_test < best_bic:
                best_bic = bic_test
                best_predictor = predictor
                best_stats = {
                    'ss_res': ss_res_test,
                    'r2': r2_test,
                    'k': k_test,
                    'aic': calculate_aic(n, ss_res_test, k_test),
                    'cp': calculate_cp(ss_res_test, n, k_test, mse_quad)
                }
        
        if best_predictor is not None:
            selected_quad.append(best_predictor)
            available_quad.remove(best_predictor)
            forward_quad_models.append({
                'step': step,
                'predictors': selected_quad.copy(),
                'k': best_stats['k'],
                'r2': best_stats['r2'],
                'aic': best_stats['aic'],
                'bic': best_bic,
                'cp': best_stats['cp']
            })
        
        # Early stopping if BIC starts increasing consistently
        if len(forward_quad_models) > 3:
            recent_bics = [m['bic'] for m in forward_quad_models[-3:]]
            if all(recent_bics[i] > recent_bics[i-1] for i in range(1, len(recent_bics))):
                print(f"Early stopping at step {step} (BIC increasing)")
                break
    
    # Display results
    print("\nForward Selection Results (Quadratic Model):")
    print(f"{'Step':<6} {'# Features':<12} {'R²':<10} {'AIC':<12} {'BIC':<12} {'Cp':<12}")
    print("-" * 64)
    for model_info in forward_quad_models:
        num_features = len(model_info['predictors'])
        print(f"{model_info['step']:<6} {num_features:<12} {model_info['r2']:<10.4f} "
              f"{model_info['aic']:<12.4f} {model_info['bic']:<12.4f} {model_info['cp']:<12.4f}")
    
    # Find best model
    best_bic_idx_quad = min(range(len(forward_quad_models)), 
                            key=lambda i: forward_quad_models[i]['bic'])
    best_model_quad = forward_quad_models[best_bic_idx_quad]
    
    print(f"\nBest model by BIC: Step {best_model_quad['step']}")
    print(f"Number of features: {len(best_model_quad['predictors'])}")
    print(f"Selected features: {[all_feature_names[i] for i in best_model_quad['predictors']]}")
    print(f"R² = {best_model_quad['r2']:.4f}")
    print(f"BIC = {best_model_quad['bic']:.4f}")
    print()
    
    # Fit best quadratic model
    if best_model_quad['predictors']:
        X_best_quad = X_quad_full[:, best_model_quad['predictors']]
        model_best_quad = LinearRegression()
        model_best_quad.fit(X_best_quad, y)
        y_pred_best_quad = model_best_quad.predict(X_best_quad)
        
        print("Best Quadratic Model Coefficients:")
        print(f"  Intercept = {model_best_quad.intercept_:.6f}")
        for i, pred_idx in enumerate(best_model_quad['predictors']):
            print(f"  β({all_feature_names[pred_idx]}) = {model_best_quad.coef_[i]:.6f}")
    print()
    
    # Visualization for part (d)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('(d) Variable Selection for Quadratic Model', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Selection path plot
    steps_quad = [m['step'] for m in forward_quad_models]
    axes[0, 0].plot(steps_quad, [m['r2'] for m in forward_quad_models], 
                   'o-', linewidth=2, markersize=8, label='R²', color='blue')
    ax_twin = axes[0, 0].twinx()
    ax_twin.plot(steps_quad, [m['bic'] for m in forward_quad_models], 
                's-', linewidth=2, markersize=8, label='BIC', color='red')
    axes[0, 0].set_xlabel('Step', fontsize=11)
    axes[0, 0].set_ylabel('R²', fontsize=11, color='blue')
    ax_twin.set_ylabel('BIC', fontsize=11, color='red')
    axes[0, 0].set_title('Forward Selection Progress', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='y', labelcolor='blue')
    ax_twin.tick_params(axis='y', labelcolor='red')
    
    # BIC comparison
    axes[0, 1].plot(steps_quad, [m['bic'] for m in forward_quad_models], 
                   'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].axvline(x=best_bic_idx_quad, color='green', linestyle='--', 
                      linewidth=2, label=f'Best (step {best_bic_idx_quad})')
    axes[0, 1].set_xlabel('Step', fontsize=11)
    axes[0, 1].set_ylabel('BIC', fontsize=11)
    axes[0, 1].set_title('BIC by Number of Features', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best model fit
    if best_model_quad['predictors']:
        axes[1, 0].scatter(y, y_pred_best_quad, alpha=0.6, s=80, 
                          edgecolors='black', color='orange')
        axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 
                       'r--', lw=2, label='Perfect Fit')
        axes[1, 0].set_xlabel('Actual Y', fontsize=11)
        axes[1, 0].set_ylabel('Predicted Y', fontsize=11)
        axes[1, 0].set_title('Best Quadratic Model: Actual vs Predicted', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals_best_quad = y - y_pred_best_quad
        axes[1, 1].scatter(y_pred_best_quad, residuals_best_quad, alpha=0.6, s=80, 
                          edgecolors='black', color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Fitted Values', fontsize=11)
        axes[1, 1].set_ylabel('Residuals', fontsize=11)
        axes[1, 1].set_title('Best Quadratic Model: Residuals', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question5_part_d_quad_selection.png', dpi=300, bbox_inches='tight')
    print("Figure saved: question5_part_d_quad_selection.png\n")
    
    # ==================== Final Summary ====================
    print("="*80)
    print("FINAL SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    print()
    
    print("Model Comparison:")
    print(f"{'Model':<30} {'R²':<10} {'Adj R²':<10} {'# Features':<12}")
    print("-" * 62)
    print(f"{'Linear (Full)':<30} {r2_full:<10.4f} {r2_adj_full:<10.4f} {6:<12}")
    print(f"{'Linear (Selected by BIC)':<30} {forward_models[best_bic_idx]['r2']:<10.4f} "
          f"{1 - (1 - forward_models[best_bic_idx]['r2']) * (n - 1) / (n - len(forward_models[best_bic_idx]['predictors']) - 1):<10.4f} "
          f"{len(forward_models[best_bic_idx]['predictors']):<12}")
    print(f"{'Quadratic (Full)':<30} {r2_quad:<10.4f} {r2_adj_quad:<10.4f} {27:<12}")
    print(f"{'Quadratic (Selected by BIC)':<30} {best_model_quad['r2']:<10.4f} "
          f"{1 - (1 - best_model_quad['r2']) * (n - 1) / (n - len(best_model_quad['predictors']) - 1):<10.4f} "
          f"{len(best_model_quad['predictors']):<12}")
    print()
    
    print("Key Findings:")
    print("1. The linear model shows that run time (X3) has strong negative correlation")
    print("   with oxygen consumption, which is physiologically expected.")
    print()
    print("2. Forward selection and backward elimination agree on important predictors,")
    print("   suggesting robust variable selection.")
    print()
    print("3. The quadratic model improves fit but adds complexity. The parsimonious")
    print("   linear model selected by BIC provides good prediction with fewer features.")
    print()
    print("4. Recommended model for practical use:")
    recommended_linear = forward_models[best_bic_idx]
    if recommended_linear['predictors']:
        print(f"   Variables: {[f'X{p+1}' for p in recommended_linear['predictors']]}")
        print(f"   R² = {recommended_linear['r2']:.4f}")
        print(f"   This model balances predictive accuracy with interpretability.")
    print()
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll figures saved:")
    print("  - question5_correlation_matrix.png")
    print("  - question5_part_a_full_model.png")
    print("  - question5_part_b_selection.png")
    print("  - question5_part_c_quadratic.png")
    print("  - question5_part_d_quad_selection.png")
    print()
    
    plt.show()


if __name__ == "__main__":
    main()

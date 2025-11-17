import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_table_image(data, title, filename, figsize=(12, 8), highlight_rows=None):
    """Create a styled table image"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data.values, 
                     colLabels=data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(len(data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Row styling
    for i in range(1, len(data) + 1):
        for j in range(len(data.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            
            # Highlight specific rows
            if highlight_rows and i in highlight_rows:
                cell.set_facecolor('#FFE699')
                cell.set_text_props(weight='bold')
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {filename}")


# ============================================================
# Table 1: Dataset Preview
# ============================================================
data_preview = pd.DataFrame({
    'Individual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [44.609, 45.313, 54.297, 59.571, 49.874, 44.811, 45.681, 49.091, 39.442, 60.055],
    'X1 (Age)': [44, 40, 44, 42, 38, 47, 40, 43, 44, 38],
    'X2 (Weight)': [89.47, 75.07, 85.84, 68.15, 89.02, 77.45, 75.98, 81.19, 81.42, 81.87],
    'X3 (Run Time)': [11.37, 10.07, 8.65, 8.17, 9.22, 11.63, 11.95, 10.85, 13.08, 8.63],
    'X4 (Rest Pulse)': [62, 62, 45, 40, 55, 58, 70, 64, 63, 48],
    'X5 (Begin Pulse)': [178, 185, 156, 166, 178, 176, 176, 162, 174, 170],
    'X6 (End Pulse)': [182, 185, 168, 172, 180, 176, 180, 170, 176, 186]
})

create_table_image(data_preview, 
                   'Table 1: Dataset Preview (First 10 Observations of 31)',
                   'table1_dataset_preview.png',
                   figsize=(14, 6))

# ============================================================
# Table 2: Descriptive Statistics
# ============================================================
desc_stats = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
    'Y': [31.00, 47.38, 5.33, 37.39, 44.96, 46.77, 50.13, 60.06],
    'X1': [31.00, 47.68, 5.21, 38.00, 44.00, 48.00, 51.00, 57.00],
    'X2': [31.00, 77.44, 8.33, 59.08, 73.20, 77.45, 82.33, 91.63],
    'X3': [31.00, 10.59, 1.39, 8.17, 9.78, 10.47, 11.27, 14.03],
    'X4': [31.00, 53.74, 8.29, 40.00, 48.00, 52.00, 58.50, 76.00],
    'X5': [31.00, 169.65, 10.25, 146.00, 163.00, 170.00, 176.00, 186.00],
    'X6': [31.00, 173.77, 9.16, 155.00, 168.00, 172.00, 180.00, 192.00]
})

create_table_image(desc_stats, 
                   'Table 2: Descriptive Statistics (n=31)',
                   'table2_descriptive_stats.png',
                   figsize=(14, 6))

# ============================================================
# Table 3: Correlation Matrix
# ============================================================
corr_matrix = pd.DataFrame({
    'Variable': ['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
    'Y': [1.0000, -0.3046, -0.1628, -0.8622, -0.3464, -0.3980, -0.2367],
    'X1': [-0.3046, 1.0000, -0.2335, 0.1887, -0.1416, -0.3379, -0.4329],
    'X2': [-0.1628, -0.2335, 1.0000, 0.1435, 0.0227, 0.1815, 0.2494],
    'X3': [-0.8622, 0.1887, 0.1435, 1.0000, 0.4005, 0.3136, 0.2261],
    'X4': [-0.3464, -0.1416, 0.0227, 0.4005, 1.0000, 0.3180, 0.2575],
    'X5': [-0.3980, -0.3379, 0.1815, 0.3136, 0.3180, 1.0000, 0.9298],
    'X6': [-0.2367, -0.4329, 0.2494, 0.2261, 0.2575, 0.9298, 1.0000]
})

create_table_image(corr_matrix, 
                   'Table 3: Correlation Matrix',
                   'table3_correlation_matrix.png',
                   figsize=(12, 6))

# ============================================================
# Table 4: Part (a) - Linear Model Coefficients
# ============================================================
linear_coef = pd.DataFrame({
    'Parameter': ['β₀ (Intercept)', 'β₁ (Age)', 'β₂ (Weight)', 'β₃ (Run Time)', 
                  'β₄ (Rest Pulse)', 'β₅ (Begin Pulse)', 'β₆ (End Pulse)'],
    'Estimate': [102.238339, -0.219916, -0.072380, -2.680516, -0.000844, -0.373164, 0.304735],
    'Interpretation': ['Baseline oxygen consumption', 
                      '↓0.22 per year increase',
                      '↓0.07 per kg increase',
                      '↓2.68 per minute increase ⭐',
                      'Negligible effect',
                      '↓0.37 per bpm increase',
                      '↑0.30 per bpm increase']
})

create_table_image(linear_coef, 
                   'Table 4: Part (a) - Multiple Linear Regression Coefficients',
                   'table4_part_a_coefficients.png',
                   figsize=(14, 6),
                   highlight_rows=[4])

# ============================================================
# Table 5: Part (a) - Model Statistics
# ============================================================
model_stats = pd.DataFrame({
    'Statistic': ['σ² (MSE)', 'RMSE', 'R²', 'Adjusted R²', 'F-statistic', 'Predictors'],
    'Value': [5.391972, 2.322062, 0.848003, 0.810004, 22.3163, 6],
    'Interpretation': ['Error variance', 
                      'Root mean squared error',
                      'Explains 84.8% of variation',
                      'Adjusted for # of predictors',
                      'Model is highly significant',
                      'Number of predictor variables']
})

create_table_image(model_stats, 
                   'Table 5: Part (a) - Model Statistics',
                   'table5_part_a_statistics.png',
                   figsize=(12, 5))

# ============================================================
# Table 6: Part (b) - Forward Selection Results
# ============================================================
forward_results = pd.DataFrame({
    'Step': [0, 1, 2, 3, 4, 5, 6],
    'Predictors': ['None', 'X3', 'X3,X1', 'X3,X1,X5', 'X3,X1,X5,X6', 
                   'X3,X1,X5,X6,X2', 'X3,X1,X5,X6,X2,X4'],
    'R²': [0.0000, 0.7434, 0.7642, 0.8111, 0.8368, 0.8480, 0.8480],
    'AIC': [104.70, 64.53, 63.91, 59.04, 56.50, 56.30, 58.30],
    'BIC': [106.13, 67.40, 68.21, 64.77, 63.67, 64.90, 68.34],
    'Cp': [128.90, 13.52, 12.22, 6.83, 4.77, 5.00, 7.00]
})

create_table_image(forward_results, 
                   'Table 6: Part (b) - Forward Selection Results',
                   'table6_forward_selection.png',
                   figsize=(14, 6),
                   highlight_rows=[5])

# ============================================================
# Table 7: Part (b) - Backward Elimination Results
# ============================================================
backward_results = pd.DataFrame({
    'Step': [0, 1, 2, 3, 4, 5, 6],
    'Predictors': ['X1,X2,X3,X4,X5,X6', 'X1,X2,X3,X5,X6', 'X1,X3,X5,X6', 
                   'X1,X3,X5', 'X1,X3', 'X3', 'None'],
    'R²': [0.8480, 0.8480, 0.8368, 0.8111, 0.7642, 0.7434, 0.0000],
    'AIC': [58.30, 56.30, 56.50, 59.04, 63.91, 64.53, 104.70],
    'BIC': [68.34, 64.90, 63.67, 64.77, 68.21, 67.40, 106.13],
    'Cp': [7.00, 5.00, 4.77, 6.83, 12.22, 13.52, 128.90]
})

create_table_image(backward_results, 
                   'Table 7: Part (b) - Backward Elimination Results',
                   'table7_backward_elimination.png',
                   figsize=(14, 6),
                   highlight_rows=[3])

# ============================================================
# Table 8: Part (b) - Best Models by Criterion
# ============================================================
best_models = pd.DataFrame({
    'Criterion': ['AIC (Forward)', 'AIC (Backward)', 
                  'BIC (Forward) ⭐', 'BIC (Backward) ⭐', 
                  'Cp (Forward)', 'Cp (Backward)'],
    'Selected Variables': ['X3, X1, X5, X6, X2', 'X1, X2, X3, X5, X6',
                          'X3, X1, X5, X6', 'X1, X3, X5, X6',
                          'All 6 variables', 'All 6 variables'],
    '# Variables': [5, 5, 4, 4, 6, 6],
    'R²': [0.8480, 0.8480, 0.8368, 0.8368, 0.8480, 0.8480],
    'Criterion Value': [56.30, 56.30, 63.67, 63.67, 5.00, 7.00]
})

create_table_image(best_models, 
                   'Table 8: Part (b) - Best Models by Selection Criterion',
                   'table8_best_models.png',
                   figsize=(14, 5),
                   highlight_rows=[3, 4])

# ============================================================
# Table 9: Part (c) - Quadratic Model Top Coefficients
# ============================================================
quad_coef = pd.DataFrame({
    'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Term': ['X3²', 'X1×X3', 'X3×X6', 'X3×X5', 'X1×X5', 
             'X1×X6', 'X2×X3', 'X3×X4', 'X1²', 'X2×X5'],
    'Coefficient': [-0.756858, -0.534989, 0.327401, -0.324001, 0.186674,
                   -0.175921, -0.129501, -0.128494, 0.119517, 0.094463],
    'Type': ['Quadratic', 'Interaction', 'Interaction', 'Interaction', 'Interaction',
            'Interaction', 'Interaction', 'Interaction', 'Quadratic', 'Interaction']
})

create_table_image(quad_coef, 
                   'Table 9: Part (c) - Top 10 Quadratic/Interaction Coefficients',
                   'table9_quadratic_coefficients.png',
                   figsize=(12, 7),
                   highlight_rows=[1, 2])

# ============================================================
# Table 10: Part (c) - Quadratic Model Statistics
# ============================================================
quad_stats = pd.DataFrame({
    'Statistic': ['Total Features', 'Linear Terms', 'Quad/Interaction Terms', 
                  'σ² (MSE)', 'RMSE', 'R²', 'Adjusted R²', 'Degrees of Freedom'],
    'Value': [27, 6, 21, 6.740942, 2.596332, 0.976247, 0.762470, 3],
    'Note': ['6 linear + 21 quad/interact', '', '', 
            'Error variance', '', 
            '97.6% variation explained!', 
            'Large penalty for complexity',
            'Very low (overfitting risk)']
})

create_table_image(quad_stats, 
                   'Table 10: Part (c) - Quadratic Model Statistics',
                   'table10_quadratic_statistics.png',
                   figsize=(14, 6),
                   highlight_rows=[6, 7, 8])

# ============================================================
# Table 11: Part (d) - Quadratic Model Selection
# ============================================================
quad_selection = pd.DataFrame({
    'Step': [0, 1, 2, 3, 4, 5],
    '# Features': [0, 1, 2, 3, 4, 5],
    'R²': [0.0000, 0.7434, 0.7982, 0.8291, 0.8390, 0.8413],
    'AIC': [104.70, 64.53, 59.08, 55.93, 56.08, 57.64],
    'BIC': [106.13, 67.40, 63.38, 61.67, 63.25, 66.24],
    'Cp': [97.30, 5.41, 0.48, -1.42, -0.67, 1.04]
})

create_table_image(quad_selection, 
                   'Table 11: Part (d) - Forward Selection for Quadratic Model',
                   'table11_quadratic_selection.png',
                   figsize=(12, 5),
                   highlight_rows=[4])

# ============================================================
# Table 12: Part (d) - Best Quadratic Model
# ============================================================
best_quad = pd.DataFrame({
    'Feature': ['X3', 'X1×X5', 'X1×X6'],
    'Coefficient': [-2.815919, -0.007239, 0.005935],
    'Type': ['Linear', 'Interaction', 'Interaction'],
    'Interpretation': ['Run time (negative effect)',
                      'Age × Begin Pulse interaction',
                      'Age × End Pulse interaction']
})

# Add model info
model_info = pd.DataFrame({
    'Feature': ['Intercept', '', '', 'Model R²', 'Model Adj R²', 'Model BIC'],
    'Coefficient': [86.551547, '', '', 0.8291, 0.8101, 61.67],
    'Type': ['', '', '', '', '', ''],
    'Interpretation': ['Baseline value', '', '', 
                      '82.9% explained', 
                      'Adjusted goodness of fit',
                      'Best by BIC criterion']
})

best_quad_full = pd.concat([best_quad, model_info], ignore_index=True)

create_table_image(best_quad_full, 
                   'Table 12: Part (d) - Best Quadratic Model (3 Features)',
                   'table12_best_quadratic_model.png',
                   figsize=(14, 6))

# ============================================================
# Table 13: Final Model Comparison
# ============================================================
final_comparison = pd.DataFrame({
    'Model': ['Linear (Full)', 'Linear (BIC Selected)', 'Linear (Minimal)', 
              'Quadratic (Full)', 'Quadratic (BIC Selected)'],
    '# Features': [6, 4, 2, 27, 3],
    'R²': [0.8480, 0.8368, 0.7642, 0.9762, 0.8291],
    'Adjusted R²': [0.8100, 0.8202, 0.7474, 0.7625, 0.8101],
    'Pros': ['Balanced', 'Good balance ⭐', 'Most interpretable', 
             'Perfect fit', 'Captures interaction'],
    'Cons': ['Some redundancy', 'Slight loss in R²', 'Lower accuracy',
            'Severe overfitting', 'Less interpretable'],
    'Recommended For': ['Standard use', 'Primary choice', 'Quick screening',
                       'Research only', 'Advanced analysis']
})

create_table_image(final_comparison, 
                   'Table 13: Final Model Comparison Summary',
                   'table13_final_comparison.png',
                   figsize=(16, 5),
                   highlight_rows=[2])

print("\n" + "="*60)
print("✓ ALL TABLES CREATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated 13 table images:")
print("  1. Dataset Preview")
print("  2. Descriptive Statistics")
print("  3. Correlation Matrix")
print("  4. Linear Model Coefficients")
print("  5. Linear Model Statistics")
print("  6. Forward Selection Results")
print("  7. Backward Elimination Results")
print("  8. Best Models by Criterion")
print("  9. Quadratic Coefficients")
print(" 10. Quadratic Model Statistics")
print(" 11. Quadratic Model Selection")
print(" 12. Best Quadratic Model")
print(" 13. Final Model Comparison")
print("\nAll tables saved as high-resolution PNG images (300 DPI)")
print("Ready for use in documents and presentations!")

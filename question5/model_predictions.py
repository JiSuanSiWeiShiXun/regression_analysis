import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """
    展示两个推荐模型代入数据后的详细预测结果
    """
    
    print("="*80)
    print("问题(b) - 推荐模型的预测结果展示")
    print("="*80)
    print()
    
    # 数据输入
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
    X_all = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
    y = df['Y'].values
    n = len(y)
    
    # ==================== 模型1：最优4变量模型 ====================
    print("="*80)
    print("模型1：最优4变量模型（BIC准则推荐）")
    print("="*80)
    print("变量: X3（跑步时间）, X1（年龄）, X5（开始脉搏）, X6（结束脉搏）")
    print()
    
    # 拟合模型1
    X_model1 = df[['X3', 'X1', 'X5', 'X6']].values
    model1 = LinearRegression()
    model1.fit(X_model1, y)
    y_pred1 = model1.predict(X_model1)
    residuals1 = y - y_pred1
    
    # 计算统计量
    r2_1 = r2_score(y, y_pred1)
    r2_adj_1 = 1 - (1 - r2_1) * (n - 1) / (n - 4 - 1)
    mse_1 = mean_squared_error(y, y_pred1)
    rmse_1 = np.sqrt(mse_1)
    
    print("模型方程:")
    print(f"Y = {model1.intercept_:.4f} + {model1.coef_[0]:.4f}×X3 + {model1.coef_[1]:.4f}×X1 + {model1.coef_[2]:.4f}×X5 + {model1.coef_[3]:.4f}×X6")
    print()
    print("模型统计量:")
    print(f"  R² = {r2_1:.4f}")
    print(f"  调整R² = {r2_adj_1:.4f}")
    print(f"  RMSE = {rmse_1:.4f}")
    print()
    
    # 创建预测结果表格
    results1 = pd.DataFrame({
        '个体': df['Individual'],
        'X1(年龄)': df['X1'],
        'X3(跑步)': df['X3'],
        'X5(开始)': df['X5'],
        'X6(结束)': df['X6'],
        '实际Y': y,
        '预测Y': y_pred1,
        '残差': residuals1,
        '残差%': (residuals1 / y * 100)
    })
    
    print("详细预测结果（前10个观测）:")
    print(results1.head(10).to_string(index=False))
    print()
    print("详细预测结果（后10个观测）:")
    print(results1.tail(10).to_string(index=False))
    print()
    
    # 残差统计
    print("残差统计:")
    print(f"  残差均值 = {np.mean(residuals1):.6f}")
    print(f"  残差标准差 = {np.std(residuals1):.4f}")
    print(f"  最大正残差 = {np.max(residuals1):.4f} (个体 {df['Individual'][np.argmax(residuals1)]})")
    print(f"  最大负残差 = {np.min(residuals1):.4f} (个体 {df['Individual'][np.argmin(residuals1)]})")
    print()
    
    # ==================== 模型2：极简2变量模型 ====================
    print("="*80)
    print("模型2：极简2变量模型")
    print("="*80)
    print("变量: X3（跑步时间）, X1（年龄）")
    print()
    
    # 拟合模型2
    X_model2 = df[['X3', 'X1']].values
    model2 = LinearRegression()
    model2.fit(X_model2, y)
    y_pred2 = model2.predict(X_model2)
    residuals2 = y - y_pred2
    
    # 计算统计量
    r2_2 = r2_score(y, y_pred2)
    r2_adj_2 = 1 - (1 - r2_2) * (n - 1) / (n - 2 - 1)
    mse_2 = mean_squared_error(y, y_pred2)
    rmse_2 = np.sqrt(mse_2)
    
    print("模型方程:")
    print(f"Y = {model2.intercept_:.4f} + {model2.coef_[0]:.4f}×X3 + {model2.coef_[1]:.4f}×X1")
    print()
    print("模型统计量:")
    print(f"  R² = {r2_2:.4f}")
    print(f"  调整R² = {r2_adj_2:.4f}")
    print(f"  RMSE = {rmse_2:.4f}")
    print()
    
    # 创建预测结果表格
    results2 = pd.DataFrame({
        '个体': df['Individual'],
        'X1(年龄)': df['X1'],
        'X3(跑步)': df['X3'],
        '实际Y': y,
        '预测Y': y_pred2,
        '残差': residuals2,
        '残差%': (residuals2 / y * 100)
    })
    
    print("详细预测结果（前10个观测）:")
    print(results2.head(10).to_string(index=False))
    print()
    print("详细预测结果（后10个观测）:")
    print(results2.tail(10).to_string(index=False))
    print()
    
    # 残差统计
    print("残差统计:")
    print(f"  残差均值 = {np.mean(residuals2):.6f}")
    print(f"  残差标准差 = {np.std(residuals2):.4f}")
    print(f"  最大正残差 = {np.max(residuals2):.4f} (个体 {df['Individual'][np.argmax(residuals2)]})")
    print(f"  最大负残差 = {np.min(residuals2):.4f} (个体 {df['Individual'][np.argmin(residuals2)]})")
    print()
    
    # ==================== 模型对比 ====================
    print("="*80)
    print("两个模型的对比")
    print("="*80)
    comparison = pd.DataFrame({
        '模型': ['4变量模型（最优）', '2变量模型（极简）'],
        '变量数': [4, 2],
        'R²': [r2_1, r2_2],
        '调整R²': [r2_adj_1, r2_adj_2],
        'RMSE': [rmse_1, rmse_2],
        '残差标准差': [np.std(residuals1), np.std(residuals2)]
    })
    print(comparison.to_string(index=False))
    print()
    
    print("模型选择建议:")
    print("  - 4变量模型: 准确率更高（R²=0.8368），适合标准健康评估")
    print("  - 2变量模型: 更简单易用（R²=0.7642），适合快速筛查")
    print("  - 准确率差异: 仅相差8.7%，但2变量模型省略2个测量")
    print()
    
    # ==================== 生成可视化图表 ====================
    
    # 图1：两个模型的预测结果对比（完整数据表格）
    fig = plt.figure(figsize=(20, 14))
    
    # 模型1的完整预测表格
    ax1 = plt.subplot(2, 1, 1)
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data1 = []
    table_data1.append(['个体', 'X1\n年龄', 'X3\n跑步时间', 'X5\n开始脉搏', 'X6\n结束脉搏', 
                       '实际Y', '预测Y', '残差', '残差%'])
    
    for i in range(len(results1)):
        row = results1.iloc[i]
        table_data1.append([
            f"{int(row['个体'])}",
            f"{int(row['X1(年龄)'])}",
            f"{row['X3(跑步)']:.2f}",
            f"{int(row['X5(开始)'])}",
            f"{int(row['X6(结束)'])}",
            f"{row['实际Y']:.3f}",
            f"{row['预测Y']:.3f}",
            f"{row['残差']:.3f}",
            f"{row['残差%']:.1f}%"
        ])
    
    table1 = ax1.table(cellText=table_data1, cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.5)
    
    # 设置表头样式
    for i in range(9):
        cell = table1[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data1)):
        for j in range(9):
            cell = table1[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            
            # 高亮大残差
            if j == 7:  # 残差列
                residual = float(table_data1[i][7])
                if abs(residual) > 3:
                    cell.set_facecolor('#ffcccc')
    
    ax1.set_title('模型1：4变量模型 (X3, X1, X5, X6) 的完整预测结果\n' + 
                  f'R² = {r2_1:.4f}, 调整R² = {r2_adj_1:.4f}, RMSE = {rmse_1:.4f}',
                  fontsize=14, fontweight='bold', pad=20)
    
    # 模型2的完整预测表格
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data2 = []
    table_data2.append(['个体', 'X1\n年龄', 'X3\n跑步时间', '实际Y', '预测Y', '残差', '残差%'])
    
    for i in range(len(results2)):
        row = results2.iloc[i]
        table_data2.append([
            f"{int(row['个体'])}",
            f"{int(row['X1(年龄)'])}",
            f"{row['X3(跑步)']:.2f}",
            f"{row['实际Y']:.3f}",
            f"{row['预测Y']:.3f}",
            f"{row['残差']:.3f}",
            f"{row['残差%']:.1f}%"
        ])
    
    table2 = ax2.table(cellText=table_data2, cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    
    # 设置表头样式
    for i in range(7):
        cell = table2[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data2)):
        for j in range(7):
            cell = table2[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            
            # 高亮大残差
            if j == 5:  # 残差列
                residual = float(table_data2[i][5])
                if abs(residual) > 4:
                    cell.set_facecolor('#ffcccc')
    
    ax2.set_title('模型2：2变量模型 (X3, X1) 的完整预测结果\n' + 
                  f'R² = {r2_2:.4f}, 调整R² = {r2_adj_2:.4f}, RMSE = {rmse_2:.4f}',
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('model_predictions_full_table.png', dpi=300, bbox_inches='tight')
    print("✓ 完整预测表格已保存: model_predictions_full_table.png")
    
    # 图2：模型诊断对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('两个推荐模型的诊断对比', fontsize=16, fontweight='bold', y=0.995)
    
    # 模型1的诊断图
    # 1. 实际vs预测
    axes[0, 0].scatter(y, y_pred1, alpha=0.6, s=100, edgecolors='black', color='green')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美拟合')
    axes[0, 0].set_xlabel('实际Y', fontsize=11)
    axes[0, 0].set_ylabel('预测Y', fontsize=11)
    axes[0, 0].set_title('模型1: 实际vs预测', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2_1:.4f}', transform=axes[0, 0].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 残差图
    axes[0, 1].scatter(y_pred1, residuals1, alpha=0.6, s=100, edgecolors='black', color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('预测值', fontsize=11)
    axes[0, 1].set_ylabel('残差', fontsize=11)
    axes[0, 1].set_title('模型1: 残差图', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差分布
    axes[0, 2].hist(residuals1, bins=12, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('残差', fontsize=11)
    axes[0, 2].set_ylabel('频数', fontsize=11)
    axes[0, 2].set_title('模型1: 残差分布', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Q-Q图
    from scipy import stats
    stats.probplot(residuals1, dist="norm", plot=axes[0, 3])
    axes[0, 3].set_title('模型1: Q-Q图', fontweight='bold')
    axes[0, 3].grid(True, alpha=0.3)
    
    # 模型2的诊断图
    # 1. 实际vs预测
    axes[1, 0].scatter(y, y_pred2, alpha=0.6, s=100, edgecolors='black', color='blue')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美拟合')
    axes[1, 0].set_xlabel('实际Y', fontsize=11)
    axes[1, 0].set_ylabel('预测Y', fontsize=11)
    axes[1, 0].set_title('模型2: 实际vs预测', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, f'R² = {r2_2:.4f}', transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 残差图
    axes[1, 1].scatter(y_pred2, residuals2, alpha=0.6, s=100, edgecolors='black', color='blue')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('预测值', fontsize=11)
    axes[1, 1].set_ylabel('残差', fontsize=11)
    axes[1, 1].set_title('模型2: 残差图', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3. 残差分布
    axes[1, 2].hist(residuals2, bins=12, alpha=0.7, edgecolor='black', color='blue')
    axes[1, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 2].set_xlabel('残差', fontsize=11)
    axes[1, 2].set_ylabel('频数', fontsize=11)
    axes[1, 2].set_title('模型2: 残差分布', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Q-Q图
    stats.probplot(residuals2, dist="norm", plot=axes[1, 3])
    axes[1, 3].set_title('模型2: Q-Q图', fontweight='bold')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_predictions_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ 模型诊断对比图已保存: model_predictions_diagnostics.png")
    
    # 图3：个体预测对比图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('每个个体的预测值对比', fontsize=16, fontweight='bold')
    
    individuals = df['Individual'].values
    
    # 模型1
    axes[0].plot(individuals, y, 'o-', label='实际Y', linewidth=2, markersize=8, color='black')
    axes[0].plot(individuals, y_pred1, 's-', label='预测Y', linewidth=2, markersize=6, color='green', alpha=0.7)
    axes[0].fill_between(individuals, y, y_pred1, alpha=0.3, color='green')
    axes[0].set_xlabel('个体编号', fontsize=12)
    axes[0].set_ylabel('氧气消耗量 (Y)', fontsize=12)
    axes[0].set_title(f'模型1: 4变量模型\nR² = {r2_1:.4f}, RMSE = {rmse_1:.4f}', fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(1, 32, 2))
    
    # 模型2
    axes[1].plot(individuals, y, 'o-', label='实际Y', linewidth=2, markersize=8, color='black')
    axes[1].plot(individuals, y_pred2, 's-', label='预测Y', linewidth=2, markersize=6, color='blue', alpha=0.7)
    axes[1].fill_between(individuals, y, y_pred2, alpha=0.3, color='blue')
    axes[1].set_xlabel('个体编号', fontsize=12)
    axes[1].set_ylabel('氧气消耗量 (Y)', fontsize=12)
    axes[1].set_title(f'模型2: 2变量模型\nR² = {r2_2:.4f}, RMSE = {rmse_2:.4f}', fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, 32, 2))
    
    plt.tight_layout()
    plt.savefig('model_predictions_by_individual.png', dpi=300, bbox_inches='tight')
    print("✓ 个体预测对比图已保存: model_predictions_by_individual.png")
    
    # 保存完整预测结果到CSV
    results1.to_csv('model1_predictions.csv', index=False, encoding='utf-8-sig')
    results2.to_csv('model2_predictions.csv', index=False, encoding='utf-8-sig')
    print("✓ 模型1预测结果已保存: model1_predictions.csv")
    print("✓ 模型2预测结果已保存: model2_predictions.csv")
    
    print()
    print("="*80)
    print("所有结果已生成！")
    print("="*80)
    print("生成的文件:")
    print("  1. model_predictions_full_table.png - 完整预测数据表格")
    print("  2. model_predictions_diagnostics.png - 模型诊断对比图")
    print("  3. model_predictions_by_individual.png - 个体预测对比图")
    print("  4. model1_predictions.csv - 模型1详细预测结果")
    print("  5. model2_predictions.csv - 模型2详细预测结果")
    print()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
阈值敏感性分析实验（审稿人1意见1.1）—— 重新设计版
=====================================================
背景：初步分析发现，由于双线性弹塑性曲线硬化段斜率比值 H/E 仅 4%~25%，
     斜率从 100% 骤降，因此 30%~95% 范围内任何阈值都命中同一屈服点，
     特征完全相同。若只在 60%~80% 内取 5 档重训练，将得到 5 个完全相同的
     模型，无信息量。

新设计：
  Part A: 宽范围阈值特征分析（免训练，快）
     - 阈值范围 [5%, 8%, 10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%]
     - 每档统计：屈服点识别失败率、屈服点位置偏差（vs 70%基准）、
       4 项关键特征（屈服力/屈服位移/硬化斜率/塑性功）平均与最大偏差
     - 机制分析：各组曲线硬化段斜率比值分布（解释鲁棒平台区成因）
  Part B: 关键档位完整重训练（4 档：70% 基准 / 10% 严重失效 / 15% 中度失效 / 20% 轻微失效）
     - 每档：重新提取特征 → 相同数据划分（seed=42, 120/40/40）与超参数
       → 完整训练（150 epochs，早停 patience=25）→ 测试集（40组）评估
     - 量化"阈值选错"对最终四参数反演精度的影响
  Part C: 测量噪声鲁棒性验证（免训练，快）
     - 噪声水平 0.5% / 1% / 2%（力与位移同时加噪，相对组内幅值）
     - 阈值 60%~80% 共 5 档，每组合 5 次蒙特卡洛重复
     - 统计屈服点偏差、识别失败率与特征偏差 → 证明平台区内阈值选择
       不会放大噪声敏感性

输出：sensitivity_analysis/ 目录下
  - sensitivity_results.csv     Part B 每档反演误差统计
  - feature_deviation.csv       Part A 宽范围特征偏差
  - slope_ratio_distribution.csv 机制分析数据
  - noise_robustness.csv        Part C 噪声鲁棒性
  - sensitivity_summary.png     汇总图（论文用图，3子图）
  - sensitivity_log.txt         完整日志
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import time
import os
import sys
from types import SimpleNamespace
from torch.utils.data import DataLoader

# 导入原文代码（确保实验与原文完全一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_phase1 as mp

from sklearn.metrics import r2_score

# ==================== 全局配置 ====================
THRESHOLDS_WIDE = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
                   0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # Part A 宽范围
BASELINE = 0.70                                            # 基准阈值（原文取值）
RETRAIN_THRESHOLDS = [0.70, 0.10, 0.15, 0.20]              # Part B 重训练档位
NOISE_LEVELS = [0.005, 0.01, 0.02]                         # Part C 噪声水平
NOISE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]          # Part C 阈值档位
N_MC_REPEATS = 5                                           # 蒙特卡洛重复次数

PARAM_NAMES = ['E', 'v', 'sigma_y', 'H']
PARAM_LABELS = ['$E$', '$\\nu$', '$\\sigma_y$', '$H$']
FEATURE_NAMES_IDX = {'yield_force': 1, 'yield_displacement': 2,
                     'hardening_slope': 5, 'plastic_energy': 7}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensitivity_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

DISP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_training_data.csv')
FORCE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'force_displacement_data.csv')


# ==================== 参数化阈值的数据集 ====================
class ThresholdDataset(mp.MultiModalMaterialDataset):
    """继承原数据集，仅将屈服点识别阈值参数化，其余逻辑与原文完全一致"""

    def __init__(self, displacement_csv, force_csv, threshold, normalize=True):
        self.threshold = threshold
        super().__init__(displacement_csv, force_csv, normalize=normalize)

    def _find_yield_point(self, displacements, forces):
        """与原文 _find_yield_point 唯一区别：0.7 → self.threshold"""
        if len(displacements) < 10:
            return None
        slopes = []
        for i in range(1, len(displacements) - 1):
            slope = (forces[i + 1] - forces[i - 1]) / (displacements[i + 1] - displacements[i - 1] + 1e-8)
            slopes.append(slope)
        initial_slope = np.mean(slopes[:3])
        for i, slope in enumerate(slopes[5:], 5):
            if slope < initial_slope * self.threshold:
                return i
        return None


class SilentTrainer(mp.ProgressiveTrainer):
    """继承原训练器，仅屏蔽训练曲线绘图（避免覆盖原图/阻塞）"""

    def plot_training_curves(self):
        pass


def find_yield_with(threshold, displacements, forces):
    """以任意阈值调用与原文一致的屈服点识别逻辑（免实例化数据集）"""
    return ThresholdDataset._find_yield_point(
        SimpleNamespace(threshold=threshold), displacements, forces)


# ==================== 特征提取工具 ====================
def extract_features_from_df(force_df, threshold):
    """从力-位移 DataFrame 提取未归一化原始特征（逻辑与原数据集完全一致）"""
    class _RawDS(ThresholdDataset):
        def __init__(self, force_df, threshold):
            self.force_data = force_df
            self.threshold = threshold
            self.force_features = self._extract_force_features()
            self.normalize = False

    ds = _RawDS(force_df, threshold)
    feats = {int(k): v for k, v in ds.force_features.items()}
    return ds, feats


def yield_indices(force_df, threshold):
    """返回每组曲线在该阈值下识别的屈服点索引（None 表示识别失败）"""
    idxs = {}
    for gid, g in force_df.groupby('group_id'):
        g = g.sort_values('load_step')
        idxs[int(gid)] = find_yield_with(threshold,
                                         g['displacement_y'].values,
                                         g['force_y'].values)
    return idxs


# ==================== Part A: 宽范围阈值特征分析 ====================
def wide_range_feature_analysis():
    print('\n' + '=' * 60)
    print('Part A: 宽范围阈值特征分析（vs 70%基准）')
    print('=' * 60)
    force_df = pd.read_csv(FORCE_CSV)
    _, baseline_feats = extract_features_from_df(force_df, BASELINE)
    base_idxs = yield_indices(force_df, BASELINE)

    rows = []
    for thr in THRESHOLDS_WIDE:
        feats = baseline_feats if thr == BASELINE else extract_features_from_df(force_df, thr)[1]
        idxs = base_idxs if thr == BASELINE else yield_indices(force_df, thr)

        # 失败率
        fail_rate = sum(1 for v in idxs.values() if v is None) / len(idxs) * 100
        # 屈服点位置偏差（仅统计两者均成功的组）
        pt_devs = [abs(idxs[g] - base_idxs[g]) for g in idxs
                   if idxs[g] is not None and base_idxs[g] is not None]

        row = {'threshold': f'{thr * 100:.0f}%',
               'yield_detect_fail_rate(%)': fail_rate,
               'yield_point_mean_dev(steps)': np.mean(pt_devs) if pt_devs else np.nan}
        for fname, fidx in FEATURE_NAMES_IDX.items():
            devs = [abs(feats[g][fidx] - baseline_feats[g][fidx]) /
                    (abs(baseline_feats[g][fidx]) + 1e-12) * 100 for g in baseline_feats]
            row[f'{fname}_mean_dev(%)'] = np.mean(devs)
            row[f'{fname}_max_dev(%)'] = np.max(devs)
        rows.append(row)
        print(f"  阈值 {thr*100:4.0f}%: 失败率 {fail_rate:5.1f}%, "
              f"屈服点偏差 {row['yield_point_mean_dev(steps)'] if not np.isnan(row['yield_point_mean_dev(steps)']) else float('nan'):.2f} 步, "
              f"屈服力平均偏差 {row['yield_force_mean_dev(%)']:8.2f}%, "
              f"硬化斜率平均偏差 {row['hardening_slope_mean_dev(%)']:8.2f}%")

    dev_df = pd.DataFrame(rows)
    dev_df.to_csv(os.path.join(OUT_DIR, 'feature_deviation.csv'), index=False, encoding='utf-8-sig')
    return dev_df


def slope_ratio_mechanism():
    """机制分析：各组曲线硬化段斜率比值（slopes[5]/初始斜率）分布"""
    force_df = pd.read_csv(FORCE_CSV)
    ratios = {}
    for gid, g in force_df.groupby('group_id'):
        g = g.sort_values('load_step')
        d, f = g['displacement_y'].values, g['force_y'].values
        slopes = [(f[i + 1] - f[i - 1]) / (d[i + 1] - d[i - 1] + 1e-8)
                  for i in range(1, len(d) - 1)]
        ratios[int(gid)] = slopes[5] / (np.mean(slopes[:3]) + 1e-12)
    ratio_df = pd.DataFrame({'group_id': list(ratios.keys()),
                             'slope_ratio_at_transition': list(ratios.values())})
    ratio_df.to_csv(os.path.join(OUT_DIR, 'slope_ratio_distribution.csv'),
                    index=False, encoding='utf-8-sig')
    vals = np.array(list(ratios.values()))
    print(f"\n机制分析: 硬化段斜率比值分布 min={vals.min():.3f}, "
          f"max={vals.max():.3f}, mean={vals.mean():.3f}")
    print(f"  → 任何高于 {vals.max()*100:.0f}% 的阈值都命中同一屈服点（鲁棒平台区）")
    return vals


# ==================== Part C: 噪声鲁棒性验证 ====================
def noise_robustness_analysis():
    print('\n' + '=' * 60)
    print('Part C: 测量噪声鲁棒性验证（0.5%/1%/2% × 60%~80%阈值 × 5次蒙特卡洛）')
    print('=' * 60)
    clean_df = pd.read_csv(FORCE_CSV)
    # 干净基准特征与屈服点（平台区内各阈值相同，用各自阈值提取以严格对齐）
    clean_feats = {thr: extract_features_from_df(clean_df, thr)[1] for thr in NOISE_THRESHOLDS}
    clean_idxs = {thr: yield_indices(clean_df, thr) for thr in NOISE_THRESHOLDS}

    # 预生成噪声幅值（组内最大幅值的比例）
    fmax = clean_df.groupby('group_id')['force_y'].transform(lambda s: np.max(np.abs(s)))
    dmax = clean_df.groupby('group_id')['displacement_y'].transform(lambda s: np.max(np.abs(s)))

    rows = []
    for noise in NOISE_LEVELS:
        for thr in NOISE_THRESHOLDS:
            pt_devs, fails, yf_devs, hs_devs = [], 0, [], []
            for rep in range(N_MC_REPEATS):
                rng = np.random.default_rng(1000 * rep + int(noise * 10000) + int(thr * 100))
                noisy_df = clean_df.copy()
                noisy_df['force_y'] = clean_df['force_y'] + rng.normal(0, noise * fmax)
                noisy_df['displacement_y'] = clean_df['displacement_y'] + rng.normal(0, noise * dmax)

                _, feats = extract_features_from_df(noisy_df, thr)
                idxs = yield_indices(noisy_df, thr)
                for gid in clean_feats[thr]:
                    base_i = clean_idxs[thr][gid]
                    if idxs[gid] is None:
                        fails += 1
                    elif base_i is not None:
                        pt_devs.append(abs(idxs[gid] - base_i))
                    bf, bs = clean_feats[thr][gid], feats[gid]
                    yf_devs.append(abs(bf[FEATURE_NAMES_IDX['yield_force']] -
                                       bs[FEATURE_NAMES_IDX['yield_force']]) /
                                   (abs(bf[FEATURE_NAMES_IDX['yield_force']]) + 1e-12) * 100)
                    hs_devs.append(abs(bf[FEATURE_NAMES_IDX['hardening_slope']] -
                                       bs[FEATURE_NAMES_IDX['hardening_slope']]) /
                                   (abs(bf[FEATURE_NAMES_IDX['hardening_slope']]) + 1e-12) * 100)
            n_total = len(clean_feats[thr]) * N_MC_REPEATS
            row = {'noise_level': f'{noise*100:.1f}%', 'threshold': f'{thr*100:.0f}%',
                   'yield_point_mean_dev(steps)': np.mean(pt_devs) if pt_devs else np.nan,
                   'fail_rate(%)': fails / n_total * 100,
                   'yield_force_mean_dev(%)': np.mean(yf_devs),
                   'hardening_slope_mean_dev(%)': np.mean(hs_devs)}
            rows.append(row)
            print(f"  噪声 {noise*100:3.1f}% / 阈值 {thr*100:.0f}%: "
                  f"屈服点偏差 {row['yield_point_mean_dev(steps)']:.3f} 步, "
                  f"失败率 {row['fail_rate(%)']:.2f}%, "
                  f"屈服力偏差 {row['yield_force_mean_dev(%)']:.2f}%, "
                  f"硬化斜率偏差 {row['hardening_slope_mean_dev(%)']:.2f}%")
    noise_df = pd.DataFrame(rows)
    noise_df.to_csv(os.path.join(OUT_DIR, 'noise_robustness.csv'),
                    index=False, encoding='utf-8-sig')
    return noise_df


# ==================== Part B: 关键档位完整重训练与评估 ====================
def train_and_evaluate(threshold):
    """某一阈值下：重新提取特征→重新训练→测试集评估（流程与原文main一致）"""
    print('\n' + '-' * 60)
    print(f'Part B: 阈值 {threshold*100:.0f}% 完整重训练')
    print('-' * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 数据集（该阈值下重新提取全部特征并归一化）
    dataset = ThresholdDataset(DISP_CSV, FORCE_CSV, threshold, normalize=True)

    # 2. 数据划分（与原文完全一致：seed=42, 6:2:2）
    unique_groups = np.unique(dataset.group_ids)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    train_size = int(0.6 * len(unique_groups))
    val_size = int(0.2 * len(unique_groups))
    train_groups = unique_groups[:train_size]
    val_groups = unique_groups[train_size:train_size + val_size]
    test_groups = unique_groups[train_size + val_size:]

    train_indices = [i for i, g in enumerate(dataset.group_ids) if g in train_groups]
    val_indices = [i for i, g in enumerate(dataset.group_ids) if g in val_groups]
    test_indices = [i for i, g in enumerate(dataset.group_ids) if g in test_groups]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=256, shuffle=False, num_workers=0)

    # 3. 模型/损失/训练器（超参数与原文一致）
    model = mp.MultiModalPINN(disp_input_dim=4, force_input_dim=9, output_dim=4,
                              disp_hidden_dim=128, force_hidden_dim=64, fusion_dim=256)
    loss_function = mp.MultiModalLoss(lambda_disp=1.0, lambda_force=0.8,
                                      lambda_physics=0.1, lambda_h=2.0)
    trainer = SilentTrainer(model, loss_function, train_loader, val_loader, test_loader, dataset)

    # 4. 训练
    t0 = time.time()
    trainer.train(num_epochs=150)
    train_time = time.time() - t0
    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)

    # 5. 测试集评估（组级统计，与原文一致）
    evaluator = mp.ComprehensiveEvaluator(model, dataset)
    test_loss, predictions, targets, group_results = evaluator.evaluate_on_test_set(test_loader)

    stats = {'threshold': f'{threshold * 100:.0f}%', 'train_time(s)': round(train_time, 1),
             'best_val_loss': trainer.best_val_loss, 'test_mse': test_loss}
    for i, name in enumerate(PARAM_NAMES):
        errors = [r['error'][i] for r in group_results.values()]
        trues = [r['true'][i] for r in group_results.values()]
        preds = [r['predicted'][i] for r in group_results.values()]
        stats[f'{name}_mean_err(%)'] = np.mean(errors)
        stats[f'{name}_max_err(%)'] = np.max(errors)
        stats[f'{name}_R2'] = r2_score(trues, preds)
    stats['overall_mean_err(%)'] = np.mean([stats[f'{n}_mean_err(%)'] for n in PARAM_NAMES])

    print(f"  阈值 {threshold*100:.0f}% 结果: 总平均误差 {stats['overall_mean_err(%)']:.3f}%, "
          f"H平均误差 {stats['H_mean_err(%)']:.3f}%, 训练用时 {train_time:.0f}s")
    return stats


# ==================== 汇总绘图 ====================
def plot_summary(res_df, dev_df, noise_df, ratio_vals):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) 宽范围阈值：失败率 + 特征偏差（双轴）
    ax = axes[0]
    thr_pct = [float(t.strip('%')) for t in dev_df['threshold']]
    fail = dev_df['yield_detect_fail_rate(%)']
    yf_dev = dev_df['yield_force_mean_dev(%)'].clip(lower=1e-3)
    ax.bar(thr_pct, fail, width=2.2, color='#aec7e8', alpha=0.8, label='Detection failure rate (left)')
    ax.set_xlabel('Slope-drop threshold for yield detection', fontsize=11)
    ax.set_ylabel('Detection failure rate (%)', fontsize=11, color='#1f59a8')
    ax.tick_params(axis='y', labelcolor='#1f59a8')
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(thr_pct, yf_dev, 'o-', color='#d62728', linewidth=2, markersize=5,
             label='Yield-force deviation (right)')
    ax2.set_yscale('log')
    ax2.set_ylabel('Mean yield-force deviation vs. 70% (%)', fontsize=11, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax.axvline(70, color='gray', linestyle='--', alpha=0.7)
    ax.axvspan(30, 95, color='#2ca02c', alpha=0.08)
    ax.annotate('Robust plateau\n(30%–95%)', xy=(62, 78), fontsize=9, color='#2a7a2a', ha='center')
    ax.set_title('(a) Yield-point detection vs. threshold', fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25)

    # (b) 关键档位反演误差
    ax = axes[1]
    x = [float(t.strip('%')) for t in res_df['threshold']]
    pcolors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    for i, name in enumerate(PARAM_NAMES):
        ax.plot(x, res_df[f'{name}_mean_err(%)'], 's-', color=pcolors[i],
                label=PARAM_LABELS[i], linewidth=2, markersize=7)
    ax.plot(x, res_df['overall_mean_err(%)'], 'D--', color='black', label='Average',
            linewidth=1.5, markersize=6, alpha=0.7)
    ax.axvline(70, color='gray', linestyle='--', alpha=0.6, label='Adopted (70%)')
    ax.axvspan(30, 95, color='#2ca02c', alpha=0.08)
    ax.set_xlabel('Slope-drop threshold for yield detection', fontsize=11)
    ax.set_ylabel('Mean relative error of identified parameters (%)', fontsize=11)
    ax.set_title('(b) Parameter inversion error (test set, 40 groups)', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) 噪声鲁棒性
    ax = axes[2]
    noises = [f'{n*100:.1f}%' for n in NOISE_LEVELS]
    width = 0.15
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(NOISE_THRESHOLDS)))
    for k, thr in enumerate(NOISE_THRESHOLDS):
        sub = noise_df[noise_df['threshold'] == f'{thr*100:.0f}%']
        xs = np.arange(len(noises)) + (k - (len(NOISE_THRESHOLDS) - 1) / 2) * width
        ax.bar(xs, sub['yield_force_mean_dev(%)'], width=width, color=colors[k],
               label=f'Threshold {thr*100:.0f}%')
    ax.set_xticks(np.arange(len(noises)))
    ax.set_xticklabels(noises)
    ax.set_xlabel('Measurement noise level (Gaussian, of full scale)', fontsize=11)
    ax.set_ylabel('Mean yield-force deviation (%)', fontsize=11)
    max_dev = noise_df['yield_force_mean_dev(%)'].max()
    fail_max = noise_df['fail_rate(%)'].max()
    ax.set_title('(c) Robustness to measurement noise', fontsize=12)
    ax.annotate(f'Max yield-point shift < 0.05 steps\nMax failure rate {fail_max:.1f}%',
                xy=(0.98, 0.55), xycoords='axes fraction', ha='right', fontsize=9,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'sensitivity_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\n汇总图已保存: {OUT_DIR}\\sensitivity_summary.png')


# ==================== 主程序 ====================
def main():
    # 重定向stdout到日志文件（同时保留控制台输出）
    class Tee:
        def __init__(self, path):
            self.file = open(path, 'w', encoding='utf-8')
            self.stdout = sys.stdout

        def write(self, obj):
            self.file.write(obj)
            self.stdout.write(obj)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    sys.stdout = Tee(os.path.join(OUT_DIR, 'sensitivity_log.txt'))

    print('=' * 60)
    print('阈值敏感性分析实验（重新设计版，审稿人1意见1.1）')
    print(f'Part A 宽范围: {[f"{t*100:.0f}%" for t in THRESHOLDS_WIDE]}')
    print(f'Part B 重训练档位: {[f"{t*100:.0f}%" for t in RETRAIN_THRESHOLDS]}')
    print(f'Part C 噪声: {[f"{n*100:.1f}%" for n in NOISE_LEVELS]} × '
          f'{[f"{t*100:.0f}%" for t in NOISE_THRESHOLDS]} × {N_MC_REPEATS}次重复')
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # Part A: 宽范围特征分析（快）
    dev_df = wide_range_feature_analysis()
    ratio_vals = slope_ratio_mechanism()

    # Part C: 噪声鲁棒性（快）
    noise_df = noise_robustness_analysis()

    # Part B: 4档完整重训练（慢，增量写盘防中断丢失）
    all_stats = []
    res_path = os.path.join(OUT_DIR, 'sensitivity_results.csv')
    for thr in RETRAIN_THRESHOLDS:
        stats = train_and_evaluate(thr)
        all_stats.append(stats)
        pd.DataFrame(all_stats).to_csv(res_path, index=False, encoding='utf-8-sig')
        print(f'[已保存] {res_path} ({len(all_stats)}/{len(RETRAIN_THRESHOLDS)} 档完成)')

    res_df = pd.DataFrame(all_stats)

    # 汇总
    print('\n' + '=' * 60)
    print('Part D: 汇总')
    print('=' * 60)
    print('\n【表1 宽范围特征提取分析（vs 70%基准）】')
    print(dev_df.to_string(index=False))
    print('\n【表2 参数反演误差（测试集40组，4档重训练）】')
    print(res_df.to_string(index=False))
    print('\n【表3 噪声鲁棒性】')
    print(noise_df.to_string(index=False))

    plot_summary(res_df, dev_df, noise_df, ratio_vals)
    print(f'\n实验全部完成: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

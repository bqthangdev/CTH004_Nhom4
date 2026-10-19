import re
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Set base style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

# Configure source and output directories robustly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == 'visualization':
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    VIS_DIR = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR
    VIS_DIR = os.path.join(PROJECT_ROOT, 'visualization')
PAPER_DIR = os.path.join(PROJECT_ROOT, 'paper')
LOG_FILE = os.path.join(PROJECT_ROOT, 'official-bipn', 'log', 'model', 'tmall_enb_64_2026-10-17 13_56_43.log')

# Output ONLY to paper/ directory as requested
OUTPUT_DIRS = [PAPER_DIR]

def save_current_figure(fn_base):
    """Saves the current matplotlib figure to all configured output directories in both PDF and PNG."""
    for out_dir in OUTPUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f'{fn_base}.pdf')
        png_path = os.path.join(out_dir, f'{fn_base}.png')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

# ==============================================================================
# 1. FIGURE 1: MULTI-METRIC PERFORMANCE COMPARISON (HR@10, NDCG@10, HR@20, NDCG@20)
# ==============================================================================
def plot_performance_metrics(lang='vi'):
    datasets = ['Tmall', 'Retailrocket']
    models_labels = {
        'vi': ['BIPN (Gốc)', 'BIPN (Có trọng số)', 'LightGCN (Đơn hành vi)'],
        'en': ['BIPN (Vanilla)', 'BIPN (Weighted)', 'LightGCN (Single-behavior)']
    }[lang]
    
    # Data: HR@10, NDCG@10, HR@20, NDCG@20
    data_hr10 = [[0.1398, 0.1275], [0.0071, 0.1229], [0.0141, 0.0003]]
    data_ndcg10 = [[0.0736, 0.0901], [0.0032, 0.0865], [0.0070, 0.0001]]
    data_hr20 = [[0.2000, 0.1515], [0.0127, 0.1482], [0.0197, 0.0007]]
    data_ndcg20 = [[0.0884, 0.0962], [0.0046, 0.0929], [0.0083, 0.0002]]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']
    width = 0.25
    x = np.arange(len(datasets))

    metrics_config = [
        (axes[0, 0], data_hr10, 'HR@10', 0.168,
         '(a) Tỷ lệ trúng đích Top 10 (HR@10)' if lang == 'vi' else '(a) Hit Ratio @ 10 (HR@10)'),
        (axes[0, 1], data_ndcg10, 'NDCG@10', 0.118,
         '(b) Độ liên quan xếp hạng Top 10 (NDCG@10)' if lang == 'vi' else '(b) Normalized Discounted Cumulative Gain @ 10 (NDCG@10)'),
        (axes[1, 0], data_hr20, 'HR@20', 0.238,
         '(c) Tỷ lệ trúng đích Top 20 (HR@20)' if lang == 'vi' else '(c) Hit Ratio @ 20 (HR@20)'),
        (axes[1, 1], data_ndcg20, 'NDCG@20', 0.128,
         '(d) Độ liên quan xếp hạng Top 20 (NDCG@20)' if lang == 'vi' else '(d) Normalized Discounted Cumulative Gain @ 20 (NDCG@20)'),
    ]

    for ax, data, ylabel, y_max, title in metrics_config:
        for i in range(3):
            offset = (i - 1) * width
            rects = ax.bar(x + offset, data[i], width, label=models_labels[i],
                           color=colors[i], edgecolor='black', linewidth=0.7, alpha=0.9)
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=12.5, fontweight='bold', pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.45)

    # Single unified figure-level legend at top for ultra-clean layout without covering bars
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=True, fontsize=12)

    supt = 'So sánh hiệu năng đa chỉ số trên hai bộ dữ liệu thực nghiệm' if lang == 'vi' else 'Multi-Metric Performance Comparison Across Datasets'
    plt.suptitle(supt, fontsize=14.5, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_current_figure(f'fig_performance_metrics_{lang}')

# ==============================================================================
# 2. FIGURE 2: TRAINING DYNAMICS & CONVERGENCE DUAL-AXIS CURVE
# ==============================================================================
def plot_convergence_dynamics(lang='vi'):
    epochs, losses, val_hit20, val_ndcg20 = [], [], [], []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                loss_match = re.search(r'epoch (\d+).*Train loss is \[([0-9.]+)\]', line)
                if loss_match:
                    epochs.append(int(loss_match.group(1)))
                    losses.append(float(loss_match.group(2)))
                val_match = re.search(r'validate (\d+).*\'hit@20\': np\.float64\(([0-9.]+)\).*\'ndcg@20\': np\.float64\(([0-9.]+)\)', line)
                if val_match:
                    val_hit20.append(float(val_match.group(2)))
                    val_ndcg20.append(float(val_match.group(3)))

    fig, ax1 = plt.subplots(figsize=(9.5, 5.0), dpi=300)
    color_loss = '#d62728'
    color_val = '#1f77b4'

    xlabel = 'Epoch (Vòng lặp huấn luyện)' if lang == 'vi' else 'Training Epoch'
    ylabel_loss = 'Hàm mất mát (Train Loss)' if lang == 'vi' else 'Training Loss'
    ylabel_val = 'Validation Hit@20' if lang == 'vi' else 'Validation Hit@20'
    label_loss = 'Train Loss'
    label_val = 'Validation Hit@20'

    ax1.set_xlabel(xlabel, fontsize=12.5, fontweight='bold')
    ax1.set_ylabel(ylabel_loss, color=color_loss, fontsize=12.5, fontweight='bold')
    line1 = ax1.plot(epochs, losses, color=color_loss, marker='o', markersize=4.5, linewidth=2.0, label=label_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlim(0.5, 41)
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel_val, color=color_val, fontsize=12.5, fontweight='bold')
    line2 = ax2.plot(epochs, val_hit20, color=color_val, marker='s', markersize=4.5, linewidth=2.0, label=label_val)
    ax2.tick_params(axis='y', labelcolor=color_val, labelsize=11)
    ax2.set_ylim(0.215, 0.325)

    best_epoch = 33
    best_hit = val_hit20[best_epoch - 1]
    ax2.scatter([best_epoch], [best_hit], color='#ff7f0e', s=120, zorder=5, edgecolor='black', linewidth=1.2)
    
    note_best = f'Điểm tối ưu: Epoch {best_epoch}\n(Hit@20 = {best_hit:.4f})' if lang == 'vi' else f'Optimal Checkpoint: Ep {best_epoch}\n(Hit@20 = {best_hit:.4f})'
    ax2.annotate(note_best,
                 xy=(best_epoch, best_hit), xytext=(best_epoch - 11.5, best_hit - 0.026),
                 arrowprops=dict(facecolor='#ff7f0e', shrink=0.08, width=1.8, headwidth=7),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff2cc', edgecolor='#d6b656', alpha=0.95))

    ax1.axvline(x=38, color='black', linestyle='--', linewidth=1.3, alpha=0.7)
    note_stop = 'Dừng sớm\n(Early Stopping\nPatience = 5)' if lang == 'vi' else 'Early Stopping\n(Patience = 5)'
    ax1.text(38.4, 0.150, note_stop, 
             ha='left', va='center', fontsize=10.5, fontweight='bold', fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eeeeee', edgecolor='gray', alpha=0.9))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(0.96, 0.32), frameon=True, fontsize=11.5)

    title = 'Đường cong hội tụ huấn luyện của mô hình BIPN trên bộ dữ liệu Tmall' if lang == 'vi' else 'BIPN Training Loss and Validation Convergence on Tmall'
    plt.title(title, fontsize=13.5, fontweight='bold', pad=12)
    plt.tight_layout()

    save_current_figure(f'fig_convergence_curve_{lang}')

# ==============================================================================
# 3. FIGURE 3: BEHAVIOR DISTRIBUTION & CONVERSION FUNNEL
# ==============================================================================
def plot_behavior_distribution(lang='vi'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=300)

    # Tmall
    tmall_labels = ['Click (1.81M)', 'Collect (204K)', 'Buy (198K)', ''] if lang == 'vi' else ['Click (1.81M)', 'Favorite (204K)', 'Purchase (198K)', '']
    tmall_sizes = [1813498, 204496, 198446, 1996]
    tmall_colors = ['#4a90e2', '#50e3c2', '#f5a623', '#d0021b']
    explode_tmall = (0, 0, 0, 0.08)

    def tmall_autopct(pct):
        return f'{pct:.1f}%' if pct > 1.0 else ''

    wedges1, texts1, autotexts1 = ax1.pie(
        tmall_sizes, labels=tmall_labels, autopct=tmall_autopct,
        startangle=140, colors=tmall_colors, explode=explode_tmall,
        textprops=dict(color='black', fontsize=11, fontweight='bold'), pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
    )
    for at in autotexts1:
        at.set_fontsize(11)
        at.set_fontweight('bold')

    # Leader line callout for ultra-rare Cart slice positioned in open lower-left region
    note_cart = 'Cart (2K, 0.1%)\n[Cực hiếm]' if lang == 'vi' else 'Cart (2K, 0.1%)\n[Ultra-rare]'
    ax1.annotate(note_cart, xy=(-0.77, 0.65), xytext=(-1.60, 0.38),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color='#d0021b', lw=1.8),
                 fontsize=11, fontweight='bold', color='#a00010',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='#d0021b', alpha=0.95))
    ax1.set_xlim(-1.75, 1.25)
    ax1.set_ylim(-1.25, 1.25)

    title1 = '(a) Tmall (2.22M tương tác)\nMất cân bằng cực độ (Click:Cart = 908:1)' if lang == 'vi' else '(a) Tmall (2.22M Interactions)\nExtreme Imbalance (Click:Cart = 908:1)'
    ax1.set_title(title1, fontsize=12.5, fontweight='bold', pad=10)

    # Retailrocket
    rr_labels = ['View (88.3K)', 'Cart (22.7K)', 'Buy (8.7K)'] if lang == 'vi' else ['View (88.3K)', 'Cart (22.7K)', 'Purchase (8.7K)']
    rr_sizes = [88304, 22716, 8666]
    rr_colors = ['#4a90e2', '#d0021b', '#f5a623']
    explode_rr = (0, 0.05, 0)

    wedges2, texts2, autotexts2 = ax2.pie(
        rr_sizes, labels=rr_labels, autopct='%1.1f%%',
        startangle=140, colors=rr_colors, explode=explode_rr,
        textprops=dict(color='black', fontsize=11, fontweight='bold'), pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
    )
    for at in autotexts2:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.25, 1.25)

    title2 = '(b) Retailrocket (120K tương tác)\nPhân bố cân bằng hơn (View:Cart = 3.9:1)' if lang == 'vi' else '(b) Retailrocket (120K Interactions)\nMore Balanced Ratio (View:Cart = 3.9:1)'
    ax2.set_title(title2, fontsize=12.5, fontweight='bold', pad=10)

    supt = 'Phân bố tỷ lệ hành vi người dùng trên hai bộ dữ liệu thực nghiệm' if lang == 'vi' else 'Behavior Interaction Distribution Across Datasets'
    plt.suptitle(supt, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_current_figure(f'fig_behavior_distribution_{lang}')

# ==============================================================================
# 4. FIGURE 4: SPARSITY RESILIENCE & PERFORMANCE RETENTION (GÓC NHÌN ĐỘ THƯA THỚT)
# ==============================================================================
def plot_sparsity_robustness(lang='vi'):
    """Analyzes the impact of extreme target sparsity: LightGCN collapses by 97.9%, while BIPN retains 91.2%."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=300)

    datasets = ['Tmall', 'Retailrocket']
    target_density = [198446 / 41738, 8666 / 11719] # 4.75 vs 0.74 purchases per user
    colors_density = ['#2ca02c', '#d62728']

    # Subplot 1: Average Target Interactions per User
    bars1 = ax1.bar(datasets, target_density, color=colors_density, width=0.45, edgecolor='black', linewidth=0.8)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f} ' + ('lần mua/user' if lang == 'vi' else 'buys/user'),
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Số tương tác mua TB / Người dùng' if lang == 'vi' else 'Avg Target Interactions / User', fontsize=12, fontweight='bold')
    title1 = '(a) Mật độ tương tác hành vi mua (Độ thưa thớt)' if lang == 'vi' else '(a) Target Behavior Density (Sparsity Level)'
    ax1.set_title(title1, fontsize=12.5, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=10.5)
    ax1.set_ylim(0, 5.8)
    ax1.grid(axis='y', linestyle='--', alpha=0.45)

    # Subplot 2: Relative Performance Retention Under Extreme Sparsity (Retailrocket vs Tmall HR@10)
    models = ['BIPN (Đa hành vi)' if lang == 'vi' else 'BIPN (Multi-behavior)',
              'LightGCN (Đơn hành vi)' if lang == 'vi' else 'LightGCN (Single-behavior)']
    retention = [ (0.1275 / 0.1398) * 100, (0.0003 / 0.0141) * 100 ]
    colors_ret = ['#1f77b4', '#7f7f7f']

    bars2 = ax2.bar(models, retention, color=colors_ret, width=0.45, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars2, retention):
        h = bar.get_height()
        ax2.annotate(f'{val:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    note_bipn = 'Bảo toàn 91.2%\nhiệu năng' if lang == 'vi' else 'Retains 91.2%\nPerformance'
    ax2.text(0, retention[0] - 18, note_bipn, ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    
    note_lgcn = 'Sụp đổ 97.9%\n(Đứt gãy)' if lang == 'vi' else 'Collapses 97.9%\n(Topology Failure)'
    ax2.text(1, retention[1] + 14, note_lgcn, ha='center', va='center', color='#d62728', fontweight='bold', fontsize=10.5,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fee', edgecolor='#d62728', alpha=0.9))

    ax2.set_ylabel('Tỷ lệ bảo toàn HR@10 (%)' if lang == 'vi' else 'HR@10 Retention Rate (%)', fontsize=12, fontweight='bold')
    title2 = '(b) Năng lực bảo toàn hiệu năng khi độ thưa tăng' if lang == 'vi' else '(b) Sparsity Resilience (Performance Retention)'
    ax2.set_title(title2, fontsize=12.5, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=11.5, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10.5)
    ax2.set_ylim(0, 118)
    ax2.grid(axis='y', linestyle='--', alpha=0.45)

    supt = 'Phân tích năng lực chống chịu độ thưa thớt dữ liệu cực đoan giữa BIPN và LightGCN' if lang == 'vi' else 'Sparsity Robustness and Retention Comparison: BIPN vs LightGCN'
    plt.suptitle(supt, fontsize=13.5, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_current_figure(f'fig_sparsity_robustness_{lang}')

# ==============================================================================
# 5. FIGURE 5: MULTIPLICATIVE GAIN & ABLATION STUDY WATERFALL
# ==============================================================================
def plot_relative_gain_ablation(lang='vi'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=300)

    # Subplot 1: Multiplicative Gain of Multi-behavior over Single-behavior (HR@10)
    datasets = ['Tmall', 'Retailrocket']
    gains = [0.1398 / 0.0141, 0.1275 / 0.0003]
    colors_gain = ['#4a90e2', '#2ca02c']

    bars1 = ax1.bar(datasets, gains, color=colors_gain, width=0.45, edgecolor='black', linewidth=0.8)
    for bar, g in zip(bars1, gains):
        h = bar.get_height()
        ax1.annotate(f'Gấp {g:.1f}x' if lang == 'vi' else f'{g:.1f}x Gain',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Hệ số vượt trội của BIPN so với LightGCN (lần)' if lang == 'vi' else 'BIPN Multiplicative Gain over LightGCN', fontsize=12, fontweight='bold')
    title1 = '(a) Mức độ gia tăng tri thức từ Đa hành vi (HR@10)' if lang == 'vi' else '(a) Multi-behavior Information Gain (HR@10)'
    ax1.set_title(title1, fontsize=12.5, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=10.5)
    ax1.set_ylim(0, 490)
    ax1.grid(axis='y', linestyle='--', alpha=0.45)

    note_rr = 'Đột biến 425 lần\ntrên tập thưa thớt' if lang == 'vi' else 'Dramatic 425x Surge\non Sparse Target'
    ax1.annotate(note_rr, xy=(1, gains[1]), xytext=(0.42, gains[1] - 85),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.8, headwidth=7),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe', edgecolor='#2ca02c', alpha=0.95))

    # Subplot 2: Ablation Study Impact of Weighted Loss Formulation vs Vanilla BIPN
    ablation_changes = [-94.92, -3.61]
    colors_abl = ['#d9534f', '#f0ad4e']

    bars2 = ax2.bar(datasets, ablation_changes, color=colors_abl, width=0.45, edgecolor='black', linewidth=0.8)
    for bar, change in zip(bars2, ablation_changes):
        h = bar.get_height()
        ax2.annotate(f'{change:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, -15 if h < 0 else 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax2.axhline(0, color='black', linewidth=0.9, linestyle='--')
    ax2.set_ylabel('Mức thay đổi HR@10 so với BIPN gốc (%)' if lang == 'vi' else 'HR@10 Change Relative to Vanilla BIPN (%)', fontsize=12, fontweight='bold')
    title2 = '(b) Ảnh hưởng của Trọng số tĩnh (Ablation Study)' if lang == 'vi' else '(b) Impact of Heuristic Weighting (Ablation)'
    ax2.set_title(title2, fontsize=12.5, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10.5)
    ax2.set_ylim(-112, 18)
    ax2.grid(axis='y', linestyle='--', alpha=0.45)

    supt = 'Phân tích hệ số gia tăng tri thức và tác động bóc tách của hàm mất mát có trọng số' if lang == 'vi' else 'Information Gain Multiplier and Heuristic Weighting Ablation Analysis'
    plt.suptitle(supt, fontsize=13.5, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_current_figure(f'fig_relative_gain_ablation_{lang}')

# ==============================================================================
# MAIN RUNNER: GENERATE ALL 5 PERSPECTIVES IN VIETNAMESE & ENGLISH
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING COMPREHENSIVE MULTI-FACET SCIENTIFIC FIGURES (VI & EN)...")
    print(f"Output directories: {OUTPUT_DIRS}")
    print("=" * 70)

    for lang in ['vi', 'en']:
        print(f"\n--> Rendering figures for language: [{lang.upper()}]...")
        plot_performance_metrics(lang=lang)
        plot_convergence_dynamics(lang=lang)
        plot_behavior_distribution(lang=lang)
        plot_sparsity_robustness(lang=lang)
        plot_relative_gain_ablation(lang=lang)
        print(f"    Done 5 figures for [{lang.upper()}].")

    # Maintain backward compatibility aliases for paper/main.tex across all output directories
    print("\n--> Creating compatibility copies for paper/main.tex...")
    for out_dir in OUTPUT_DIRS:
        shutil.copyfile(os.path.join(out_dir, 'fig_performance_metrics_vi.pdf'), os.path.join(out_dir, 'fig_performance_comparison.pdf'))
        shutil.copyfile(os.path.join(out_dir, 'fig_performance_metrics_vi.png'), os.path.join(out_dir, 'fig_performance_comparison.png'))
        shutil.copyfile(os.path.join(out_dir, 'fig_convergence_curve_vi.pdf'), os.path.join(out_dir, 'fig_convergence_curve.pdf'))
        shutil.copyfile(os.path.join(out_dir, 'fig_convergence_curve_vi.png'), os.path.join(out_dir, 'fig_convergence_curve.png'))
        shutil.copyfile(os.path.join(out_dir, 'fig_behavior_distribution_vi.pdf'), os.path.join(out_dir, 'fig_behavior_distribution.pdf'))
        shutil.copyfile(os.path.join(out_dir, 'fig_behavior_distribution_vi.png'), os.path.join(out_dir, 'fig_behavior_distribution.png'))

    print("=" * 70)
    print("SUCCESS: Figures synchronized across visualization/ and paper/!")
    print("=" * 70)

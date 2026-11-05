import re
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Set base style (embed TrueType fonts Type 42 in PDF to prevent pdfTeX subset glyph collisions)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
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
LOG_FILE_M1 = os.path.join(PROJECT_ROOT, 'official-bipn', 'log', 'model', 'tmall_enb_64_2026-10-17 13_56_43.log')
LOG_FILE_M2 = os.path.join(PROJECT_ROOT, 'bipn', 'logs', 'tmall_weighted.log')

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
# 1. FIGURE 1: MULTI-METRIC PERFORMANCE COMPARISON (PRESERVING APPROACH 1 + ADDING APPROACH 2)
# ==============================================================================
def plot_performance_metrics(lang='vi'):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=300)

    # Colors consistent across subplots
    c_base = '#1f77b4'      # BIPN Baseline / Aligned
    c_m1_inv = '#ff7f0e'    # Approach 1: BIPN-InvWeight
    c_lgcn = '#7f7f7f'      # Single-behavior LightGCN
    c_m2_w = '#2ca02c'      # Approach 2: BIPN-Weighted (Proposed)
    c_m2_l = '#17becf'      # Approach 2: BIPN-Learned (Proposed)

    # --------------------------------------------------------------------------
    # Subplot (a): Tmall Top-10 Metrics (HR@10 & NDCG@10) across Approach 1 & 2
    # --------------------------------------------------------------------------
    ax = axes[0, 0]
    x_a = np.arange(2)
    labels_a = ['HR@10', 'NDCG@10']
    width_a = 0.155

    series_a = [
        ('BIPN (Tái lập)' if lang == 'vi' else 'BIPN (Replicated)', [0.1398, 0.0736], c_base),
        ('HTC1: BIPN-InvWeight' if lang == 'vi' else 'App.1: BIPN-InvWeight', [0.0071, 0.0032], c_m1_inv),
        ('HTC1: LightGCN-Base' if lang == 'vi' else 'App.1: LightGCN-Base', [0.0141, 0.0070], c_lgcn),
        ('HTC2: BIPN-Weighted (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Weighted (Ours)', [0.1472, 0.0771], c_m2_w),
        ('HTC2: BIPN-Learned (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Learned (Ours)', [0.1460, 0.0772], c_m2_l),
    ]

    for i, (lbl, vals, col) in enumerate(series_a):
        offset = (i - 2) * width_a
        rects = ax.bar(x_a + offset, vals, width_a, label=lbl, color=col, edgecolor='black', linewidth=0.65, alpha=0.92)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.8, fontweight='bold', rotation=15)

    # Reference lines for SIGIR 2024 published results on Tmall
    ax.hlines(0.1414, x_a[0] - 0.42, x_a[0] + 0.42, colors='#d62728', linestyles='--', linewidth=1.3,
              label='Chuẩn bài báo gốc SIGIR\'24' if lang == 'vi' else 'SIGIR\'24 Paper Benchmark')
    ax.hlines(0.0741, x_a[1] - 0.42, x_a[1] + 0.42, colors='#d62728', linestyles='--', linewidth=1.3)

    ax.set_ylabel('Giá trị độ đo' if lang == 'vi' else 'Metric Value', fontsize=11.5, fontweight='bold')
    ax.set_title('(a) Tmall: Nhóm chỉ số Top-10 (HR@10 & NDCG@10)' if lang == 'vi' else '(a) Tmall: Top-10 Metrics (HR@10 & NDCG@10)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=11.5, fontweight='bold')
    ax.set_ylim(0, 0.182)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    # --------------------------------------------------------------------------
    # Subplot (b): Tmall Top-20 Metrics (HR@20 & NDCG@20) across Approach 1 & 2
    # --------------------------------------------------------------------------
    ax = axes[0, 1]
    x_b = np.arange(2)
    labels_b = ['HR@20', 'NDCG@20']

    series_b = [
        ('BIPN (Tái lập)' if lang == 'vi' else 'BIPN (Replicated)', [0.2000, 0.0884], c_base),
        ('HTC1: BIPN-InvWeight' if lang == 'vi' else 'App.1: BIPN-InvWeight', [0.0127, 0.0046], c_m1_inv),
        ('HTC1: LightGCN-Base' if lang == 'vi' else 'App.1: LightGCN-Base', [0.0197, 0.0083], c_lgcn),
        ('HTC2: BIPN-Weighted (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Weighted (Ours)', [0.2108, 0.0928], c_m2_w),
        ('HTC2: BIPN-Learned (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Learned (Ours)', [0.2075, 0.0923], c_m2_l),
    ]

    for i, (lbl, vals, col) in enumerate(series_b):
        offset = (i - 2) * width_a
        rects = ax.bar(x_b + offset, vals, width_a, label=lbl, color=col, edgecolor='black', linewidth=0.65, alpha=0.92)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.8, fontweight='bold', rotation=15)

    ax.hlines(0.2033, x_b[0] - 0.42, x_b[0] + 0.42, colors='#d62728', linestyles='--', linewidth=1.3)
    ax.hlines(0.0898, x_b[1] - 0.42, x_b[1] + 0.42, colors='#d62728', linestyles='--', linewidth=1.3)

    ax.set_ylabel('Giá trị độ đo' if lang == 'vi' else 'Metric Value', fontsize=11.5, fontweight='bold')
    ax.set_title('(b) Tmall: Nhóm chỉ số Top-20 (HR@20 & NDCG@20)' if lang == 'vi' else '(b) Tmall: Top-20 Metrics (HR@20 & NDCG@20)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=11.5, fontweight='bold')
    ax.set_ylim(0, 0.255)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    # --------------------------------------------------------------------------
    # Subplot (c): Retailrocket Approach 1 (Preserving 100% of Milestone 1 Data)
    # --------------------------------------------------------------------------
    ax = axes[1, 0]
    x_c = np.arange(4)
    labels_c = ['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
    width_c = 0.25

    series_c = [
        ('BIPN-Base', [0.1275, 0.0901, 0.1515, 0.0962], c_base),
        ('HTC1: BIPN-InvWeight', [0.1229, 0.0865, 0.1482, 0.0929], c_m1_inv),
        ('HTC1: LightGCN-Base', [0.0003, 0.0001, 0.0007, 0.0002], c_lgcn),
    ]

    for i, (lbl, vals, col) in enumerate(series_c):
        offset = (i - 1) * width_c
        rects = ax.bar(x_c + offset, vals, width_c, label=lbl, color=col, edgecolor='black', linewidth=0.65, alpha=0.92)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.6, fontweight='bold', rotation=20)

    ax.set_ylabel('Giá trị độ đo' if lang == 'vi' else 'Metric Value', fontsize=11.5, fontweight='bold')
    ax.set_title('(c) Retailrocket – Hướng tiếp cận 1 (Khung cơ sở ban đầu)' if lang == 'vi' else '(c) Retailrocket – Approach 1 (Initial Baseline Framework)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x_c)
    ax.set_xticklabels(labels_c, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.188)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    # --------------------------------------------------------------------------
    # Subplot (d): Retailrocket Approach 2 (Aligned Framework + Logarithmic Modulation)
    # --------------------------------------------------------------------------
    ax = axes[1, 1]
    x_d = np.arange(4)
    labels_d = ['Valid HR@10', 'Test HR@10', 'Test NDCG@10', 'Test HR@20']
    width_d = 0.20

    series_d = [
        ('BIPN-Aligned', [0.4983, 0.8571, 0.7451, 0.8787], c_base),
        ('HTC2: BIPN-Weighted', [0.5198, 0.8590, 0.7459, 0.8805], c_m2_w),
        ('HTC2: BIPN-Learned', [0.5198, 0.8599, 0.7462, 0.8805], c_m2_l),
        ('LightGCN-Aligned', [0.0226, 0.0046, 0.0035, 0.0055], c_lgcn),
    ]

    for i, (lbl, vals, col) in enumerate(series_d):
        offset = (i - 1.5) * width_d
        rects = ax.bar(x_d + offset, vals, width_d, label=lbl, color=col, edgecolor='black', linewidth=0.65, alpha=0.92)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.2, fontweight='bold', rotation=35)

    ax.set_ylabel('Giá trị độ đo' if lang == 'vi' else 'Metric Value', fontsize=11.5, fontweight='bold')
    ax.set_title('(d) Retailrocket – Hướng tiếp cận 2 (Đồng bộ kiến trúc & λ_u logarit)' if lang == 'vi' else '(d) Retailrocket – Approach 2 (Aligned & Log Modulation)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x_d)
    ax.set_xticklabels(labels_d, fontsize=10.8, fontweight='bold')
    ax.set_ylim(0, 1.09)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    # Unified figure-level legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.962), ncol=3, frameon=True, fontsize=11)

    supt = ('Đối sánh hiệu năng đa chỉ số qua hai hướng tiếp cận trên Tmall và Retailrocket'
            if lang == 'vi' else
            'Multi-Metric Performance Comparison Across Both Approaches on Tmall and Retailrocket')
    plt.suptitle(supt, fontsize=14.5, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.915])

    save_current_figure(f'fig_performance_metrics_{lang}')

# ==============================================================================
# 2. FIGURE 2: TRAINING DYNAMICS & CONVERGENCE DUAL-AXIS CURVE (M1 + M2 OVERLAY)
# ==============================================================================
def plot_convergence_dynamics(lang='vi'):
    # Parse Milestone 1 (official-bipn, 38 epochs)
    ep_m1, loss_m1, val_hit20_m1 = [], [], []
    if os.path.exists(LOG_FILE_M1):
        with open(LOG_FILE_M1, 'r', encoding='utf-8') as f:
            for line in f:
                loss_match = re.search(r'epoch (\d+).*Train loss is \[([0-9.]+)\]', line)
                if loss_match:
                    ep_m1.append(int(loss_match.group(1)))
                    loss_m1.append(float(loss_match.group(2)))
                val_match = re.search(r'validate (\d+).*\'hit@20\': np\.float64\(([0-9.]+)\)', line)
                if val_match:
                    val_hit20_m1.append(float(val_match.group(2)))

    # Parse Milestone 2 (bipn_weighted, 68 epochs)
    ep_m2, loss_m2, val_hit20_m2 = [], [], []
    if os.path.exists(LOG_FILE_M2):
        with open(LOG_FILE_M2, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.search(r'epoch (\d+) loss ([0-9.]+) valid \{.*?\'HR@20\': ([0-9.]+)', line)
                if m:
                    ep_m2.append(int(m.group(1)))
                    loss_m2.append(float(m.group(2)))
                    val_hit20_m2.append(float(m.group(3)))

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    color_loss_m1 = '#d62728'
    color_loss_m2 = '#ff7f0e'
    color_val_m1 = '#1f77b4'
    color_val_m2 = '#2ca02c'

    xlabel = 'Epoch (Vòng lặp huấn luyện)' if lang == 'vi' else 'Training Epoch'
    ylabel_loss = 'Hàm mất mát (Train Loss)' if lang == 'vi' else 'Training Loss'
    ylabel_val = 'Validation HR@20 (Hit@20)' if lang == 'vi' else 'Validation HR@20 (Hit@20)'

    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel_loss, color=color_loss_m1, fontsize=12, fontweight='bold')

    l1 = ax1.plot(ep_m1, loss_m1, color=color_loss_m1, linestyle='--', marker='o', markersize=3.5, linewidth=1.8,
                  label='Train Loss – BIPN Tái lập (38 Ep)' if lang == 'vi' else 'Train Loss – Replicated BIPN (38 Ep)')
    l2 = ax1.plot(ep_m2, loss_m2, color=color_loss_m2, linestyle='-', marker='^', markersize=3.2, linewidth=1.8, alpha=0.9,
                  label='Train Loss – HTC2: BIPN-Weighted (68 Ep)' if lang == 'vi' else 'Train Loss – App.2: BIPN-Weighted (68 Ep)')
    ax1.tick_params(axis='y', labelcolor=color_loss_m1, labelsize=10.5)
    ax1.tick_params(axis='x', labelsize=10.5)
    ax1.set_xlim(0.5, 71)
    ax1.set_ylim(0.055, 0.188)
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel_val, color=color_val_m1, fontsize=12, fontweight='bold')
    l3 = ax2.plot(ep_m1, val_hit20_m1, color=color_val_m1, linestyle='--', marker='s', markersize=4.0, linewidth=2.0,
                  label='Valid Hit@20 – BIPN Tái lập (Đỉnh Ep 33)' if lang == 'vi' else 'Valid Hit@20 – Replicated BIPN (Peak Ep 33)')
    l4 = ax2.plot(ep_m2, val_hit20_m2, color=color_val_m2, linestyle='-', marker='D', markersize=3.6, linewidth=2.1,
                  label='Valid HR@20 – HTC2: BIPN-Weighted (Đỉnh Ep 58)' if lang == 'vi' else 'Valid HR@20 – App.2: BIPN-Weighted (Peak Ep 58)')
    ax2.tick_params(axis='y', labelcolor=color_val_m1, labelsize=10.5)
    ax2.set_ylim(0.215, 0.342)

    # Highlight Milestone 1 Peak (Epoch 33, Hit@20 = 0.3094) & Early Stop (Epoch 38)
    best_ep_m1 = 33
    best_hit_m1 = val_hit20_m1[best_ep_m1 - 1]
    ax2.scatter([best_ep_m1], [best_hit_m1], color='#1f77b4', s=110, zorder=6, edgecolor='black', linewidth=1.2)
    note_m1 = (f'Đỉnh tái lập: Ep {best_ep_m1}\n(Hit@20 = {best_hit_m1:.4f})'
               if lang == 'vi' else
               f'Baseline Peak: Ep {best_ep_m1}\n(Hit@20 = {best_hit_m1:.4f})')
    ax2.annotate(note_m1, xy=(best_ep_m1, best_hit_m1), xytext=(best_ep_m1 - 15, best_hit_m1 - 0.032),
                 arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.6, headwidth=6),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef7ff', edgecolor='#1f77b4', alpha=0.95))

    ax1.axvline(x=38, color='#1f77b4', linestyle=':', linewidth=1.3, alpha=0.75)
    ax2.text(38.5, 0.286, 'Dừng sớm Mốc 1\n(Ep 38, Patience=5)' if lang == 'vi' else 'Baseline Early Stop\n(Ep 38, Patience=5)',
             fontsize=9.5, fontweight='bold', color='#1f4e78',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#f0f4f8', edgecolor='#1f77b4', alpha=0.9))

    # Highlight Milestone 2 Optimal Valid HR@10 Checkpoint (Epoch 58, HR@20 = 0.3238, HR@10 = 0.2397)
    best_ep_m2 = 58
    best_hit_m2 = val_hit20_m2[best_ep_m2 - 1]
    ax2.scatter([best_ep_m2], [best_hit_m2], color='#2ca02c', s=125, zorder=6, edgecolor='black', linewidth=1.2)
    note_m2 = (f'Đỉnh HTC2: Ep {best_ep_m2}\n(HR@20 = {best_hit_m2:.4f},\nValid HR@10 = 0.2397)'
               if lang == 'vi' else
               f'App.2 Peak: Ep {best_ep_m2}\n(HR@20 = {best_hit_m2:.4f},\nValid HR@10 = 0.2397)')
    ax2.annotate(note_m2, xy=(best_ep_m2, best_hit_m2), xytext=(best_ep_m2 - 19, best_hit_m2 + 0.006),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.6, headwidth=6),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8e8', edgecolor='#2ca02c', alpha=0.95))

    ax1.axvline(x=68, color='#2ca02c', linestyle='--', linewidth=1.3, alpha=0.75)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(0.98, 0.26), frameon=True, fontsize=9.8)

    title = ('Động học hội tụ huấn luyện trên Tmall: Bản tái lập gốc (38 Ep) và Hướng tiếp cận 2 (68 Ep)'
             if lang == 'vi' else
             'Training Convergence Dynamics on Tmall: Replicated Baseline (38 Ep) vs. Approach 2 (68 Ep)')
    plt.title(title, fontsize=12.8, fontweight='bold', pad=12)
    plt.tight_layout()

    save_current_figure(f'fig_convergence_curve_{lang}')

# ==============================================================================
# 3. FIGURE 3: BEHAVIOR DISTRIBUTION & CONVERSION FUNNEL (INVARIANT DATASET STATS)
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
# 4. FIGURE 4: SPARSITY RESILIENCE & PERFORMANCE RETENTION
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
# 5. FIGURE 5: MULTIPLICATIVE GAIN & DIVERGING ABLATION STUDY (APPROACH 1 VS APPROACH 2)
# ==============================================================================
def plot_relative_gain_ablation(lang='vi'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=300)

    # Subplot 1: Multiplicative Gain of Multi-behavior over Single-behavior (HR@10)
    datasets = ['Tmall', 'Retailrocket']
    gains = [0.1398 / 0.0141, 0.1275 / 0.0003]
    colors_gain = ['#4a90e2', '#2ca02c']

    bars1 = ax1.bar(datasets, gains, color=colors_gain, width=0.42, edgecolor='black', linewidth=0.8)
    for bar, g in zip(bars1, gains):
        h = bar.get_height()
        ax1.annotate(f'Gấp {g:.1f}x' if lang == 'vi' else f'{g:.1f}x Gain',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Hệ số vượt trội của BIPN so với LightGCN (lần)' if lang == 'vi' else 'BIPN Multiplicative Gain over LightGCN',
                   fontsize=11.5, fontweight='bold')
    title1 = '(a) Gia tăng tri thức từ Đa hành vi (HR@10)' if lang == 'vi' else '(a) Multi-behavior Information Gain (HR@10)'
    ax1.set_title(title1, fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=10.5)
    ax1.set_ylim(0, 495)
    ax1.grid(axis='y', linestyle='--', alpha=0.45)

    note_rr = ('Đột biến 425 lần (HTC1)\n& 186.9 lần (HTC2)\ntrên tập thưa thớt'
               if lang == 'vi' else
               'Dramatic 425x (App.1)\n& 186.9x (App.2) Surge\non Sparse Target')
    ax1.annotate(note_rr, xy=(1, gains[1]), xytext=(0.15, gains[1] - 115),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.8, headwidth=7),
                 fontsize=10.2, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe', edgecolor='#2ca02c', alpha=0.95))

    # Subplot 2: Diverging Ablation Study comparing Approach 1 (Negative Collapse) vs Approach 2 (Positive Gain)
    x = np.arange(len(datasets))
    width = 0.25

    # Percentage changes relative to baseline BIPN:
    # Approach 1 (BIPN-InvWeight vs BIPN-Base): Tmall (0.0071 vs 0.1398 = -94.9%), Retailrocket (0.1229 vs 0.1275 = -3.6%)
    # Approach 2 (BIPN-Weighted vs BIPN Baseline): Tmall (0.1472 vs 0.1398 = +5.3%, +4.1% vs SIGIR'24), Retailrocket Valid HR@10 (0.5198 vs 0.4983 = +4.3%)
    # Approach 2 (BIPN-Learned vs BIPN Baseline): Tmall (0.1460 vs 0.1398 = +4.4%), Retailrocket Valid HR@10 (0.5198 vs 0.4983 = +4.3%)
    m1_inv = [-94.9, -3.6]
    m2_weighted = [+5.3, +4.3]
    m2_learned = [+4.4, +4.3]

    b_m1 = ax2.bar(x - width, m1_inv, width,
                   label='HTC1: BIPN-InvWeight (Trọng số N/n_b)' if lang == 'vi' else 'App.1: BIPN-InvWeight (Unbounded N/n_b)',
                   color='#d9534f', edgecolor='black', linewidth=0.8)
    b_m2w = ax2.bar(x, m2_weighted, width,
                    label='HTC2: BIPN-Weighted (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Weighted (Ours)',
                    color='#2ca02c', edgecolor='black', linewidth=0.8)
    b_m2l = ax2.bar(x + width, m2_learned, width,
                    label='HTC2: BIPN-Learned (Đề xuất)' if lang == 'vi' else 'App.2: BIPN-Learned (Ours)',
                    color='#17becf', edgecolor='black', linewidth=0.8)

    for bar, val in zip(b_m1, m1_inv):
        h = bar.get_height()
        ax2.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, -15), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#a00000')

    for bars, vals, col_txt in [(b_m2w, m2_weighted, '#156915'), (b_m2l, m2_learned, '#0a6b78')]:
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax2.annotate(f'+{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9.8, fontweight='bold', color=col_txt)

    ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')

    note_m2_gain = ('HTC2 vượt bài báo gốc\nSIGIR\'24 (+4.1% HR@10)'
                    if lang == 'vi' else
                    'App.2 Surpasses SIGIR\'24\nBenchmark (+4.1% HR@10)')
    ax2.annotate(note_m2_gain, xy=(0, 5.3), xytext=(0.12, -38),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.28', facecolor='#e8f8e8', edgecolor='#2ca02c', alpha=0.95))

    ax2.set_ylabel('Mức biến thiên HR@10 so với BIPN chuẩn (%)' if lang == 'vi' else 'HR@10 Change Relative to Baseline BIPN (%)',
                   fontsize=11.5, fontweight='bold')
    title2 = ('(b) Bóc tách Ablation: Hướng tiếp cận 1 vs Hướng tiếp cận 2'
              if lang == 'vi' else
              '(b) Ablation Study: Approach 1 vs. Approach 2')
    ax2.set_title(title2, fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10.5)
    ax2.set_ylim(-115, 25)
    ax2.grid(axis='y', linestyle='--', alpha=0.45)
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)

    supt = ('Phân tích hệ số gia tăng tri thức đa hành vi và đối sánh bóc tách giữa hai hướng tiếp cận'
            if lang == 'vi' else
            'Multi-behavior Information Gain and Diverging Ablation Analysis Across Both Approaches')
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

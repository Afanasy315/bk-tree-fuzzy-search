import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

CSV_FILE = "bk_benchmark.csv"

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found. Run bk_benchmark first.")
    exit(1)

df = pd.read_csv(CSV_FILE)
print("Loaded data:")
print(df.to_string())

sizes = sorted(df["N"].unique())
k_vals = [1, 2, 3]

colors = {1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

# ===== CHART 1: Build time vs N =====
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["N"], df["build_time_us"] / 1000, "o-", color="#1f77b4", linewidth=2, markersize=8)
ax.set_xscale("log")
ax.set_xlabel("Dictionary size N", fontsize=13)
ax.set_ylabel("Build time (ms)", fontsize=13)
ax.set_title("BK-Tree Construction Time", fontsize=14, fontweight="bold")
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_build_time.png", dpi=150)
plt.close()
print("Saved chart1_build_time.png")

# ===== CHART 2: Search time vs N =====
fig, ax = plt.subplots(figsize=(10, 6))

# Вычисляем эмпирический alpha для каждой кривой (по последним двум точкам)
empirical_alpha = {}
theoretical_alpha = {}

for k in k_vals:
    times = df[f"bk_k{k}_time_us"].values
    # Эмпирический alpha: от N=50000 до N=100000
    t1 = df[df["N"] == 50000][f"bk_k{k}_time_us"].values[0]
    t2 = df[df["N"] == 100000][f"bk_k{k}_time_us"].values[0]
    n1 = 50000
    n2 = 100000
    alpha_emp = np.log(t2 / t1) / np.log(n2 / n1)
    empirical_alpha[k] = alpha_emp

    # Теоретический alpha по формуле α = ln(min(b, 2k+1)) / ln(b)
    b = df["avg_branching"].mean()  # среднее ветвление ~1.99
    m = min(b, 2*k + 1)
    alpha_theor = np.log(m) / np.log(b) if b > 1 else 1
    theoretical_alpha[k] = alpha_theor

    ax.plot(df["N"], times, "o-", color=colors[k], linewidth=2, markersize=7,
            label=f"BK-tree k={k} (real α={alpha_emp:.3f}, theor α={alpha_theor:.3f})")

# Full scan
fs_times = df["naive_time_us"].values
t1_fs = df[df["N"] == 50000]["naive_time_us"].values[0]
t2_fs = df[df["N"] == 100000]["naive_time_us"].values[0]
alpha_fs = np.log(t2_fs / t1_fs) / np.log(100000 / 50000)
ax.plot(df["N"], fs_times, "s--", color="black", linewidth=2, markersize=8,
        label=f"Full scan (α={alpha_fs:.3f})")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Dictionary size N", fontsize=13)
ax.set_ylabel("Search time (μs per query)", fontsize=13)
ax.set_title("Search Time vs Dictionary Size", fontsize=14, fontweight="bold")
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("chart2_search_time.png", dpi=150)
plt.close()
print("Saved chart2_search_time.png")

# ===== CHART 3: Speedup vs N =====
fig, ax = plt.subplots(figsize=(9, 5))

for k in k_vals:
    ax.plot(df["N"], df[f"bk_k{k}_speedup"], "o-", color=colors[k], linewidth=2, markersize=7, label=f"k={k}")

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.2, label="speedup = 1")
ax.set_xscale("log")
ax.set_xlabel("Dictionary size N", fontsize=13)
ax.set_ylabel("Speedup (full scan / BK-tree)", fontsize=13)
ax.set_title("BK-Tree Speedup vs Dictionary Size", fontsize=14, fontweight="bold")
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("chart3_speedup.png", dpi=150)
plt.close()
print("Saved chart3_speedup.png")

# ===== CHART 4: Search time vs k at N=100000 =====
N_FIXED = 100000
row = df[df["N"] == N_FIXED]
if row.empty:
    N_FIXED = sizes[-1]
    row = df[df["N"] == N_FIXED]

if not row.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = [0, 1, 2, 3]
    times = [
        row["bk_k0_time_us"].values[0],
        row["bk_k1_time_us"].values[0],
        row["bk_k2_time_us"].values[0],
        row["bk_k3_time_us"].values[0]
    ]
    colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax.bar([str(k) for k in k_values], times, color=colors_bar, width=0.5, edgecolor="black")
    fs_val = row["naive_time_us"].values[0]
    ax.axhline(y=fs_val, color="black", linestyle="--", linewidth=2, label=f"Full scan = {fs_val:.0f} μs")

    ax.set_xlabel("Tolerance k", fontsize=13)
    ax.set_ylabel("Search time (μs per query)", fontsize=13)
    ax.set_title(f"Search Time vs k (N = {N_FIXED:,})", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("chart4_time_vs_k.png", dpi=150)
    plt.close()
    print("Saved chart4_time_vs_k.png")
else:
    print(f"No data for N={N_FIXED}, skipping chart 4")

# ===== CHART 5: Comparison of empirical vs theoretical alpha =====
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(k_vals))
width = 0.35

empirical = [empirical_alpha[k] for k in k_vals]
theoretical = [theoretical_alpha[k] for k in k_vals]

bars1 = ax.bar(x - width/2, empirical, width, label="Empirical α", color="#ff7f0e", edgecolor="black")
bars2 = ax.bar(x + width/2, theoretical, width, label="Theoretical α", color="#1f77b4", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in k_vals], fontsize=12)
ax.set_ylabel("α (exponent)", fontsize=13)
ax.set_title("Empirical vs Theoretical α", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Подписываем значения на столбцах
for bar, val in zip(bars1, empirical):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)

for bar, val in zip(bars2, theoretical):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("chart5_alpha_comparison.png", dpi=150)
plt.close()
print("Saved chart5_alpha_comparison.png")

# ===== CONSOLE SUMMARY =====
print("\n" + "="*70)
print("ALPHA COMPARISON (from N=50000 to 100000)")
print("="*70)
print(f"Average branching b = {df['avg_branching'].mean():.4f}")
print(f"\n{'k':>4} | {'Empirical α':>12} | {'Theoretical α':>12} | {'Difference':>10}")
print("-"*50)
for k in k_vals:
    diff = empirical_alpha[k] - theoretical_alpha[k]
    print(f"{k:>4} | {empirical_alpha[k]:>12.4f} | {theoretical_alpha[k]:>12.4f} | {diff:>10.4f}")
print(f"\nFull scan α = {alpha_fs:.4f}")

print("\n" + "="*60)
print("SPEEDUP SUMMARY")
print("="*60)
print(f"{'N':>8} | {'k=1 speedup':>12} | {'k=2 speedup':>12} | {'k=3 speedup':>12}")
print("-"*50)
for _, row in df.iterrows():
    print(f"{row['N']:>8} | {row['bk_k1_speedup']:>12.2f} | {row['bk_k2_speedup']:>12.2f} | {row['bk_k3_speedup']:>12.2f}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

max_sp = max(df["bk_k1_speedup"].max(), df["bk_k2_speedup"].max(), df["bk_k3_speedup"].max())
print(f"\nMax speedup: {max_sp:.2f}x at k=1, N={df[df['bk_k1_speedup'] == max_sp]['N'].values[0]}")

print("\nWhy speedup decreases as k grows?")
print("  - k=1: check ~3 children per node")
print("  - k=3: check ~7 children per node")
print("  - Wider range [d-k, d+k] → fewer branches pruned")
print("  - At k=3, BK-tree visits almost all nodes + overhead → slower than full scan")

print("\nAll charts saved:")
print("  chart1_build_time.png")
print("  chart2_search_time.png")
print("  chart3_speedup.png")
print("  chart4_time_vs_k.png")
print("  chart5_alpha_comparison.png")

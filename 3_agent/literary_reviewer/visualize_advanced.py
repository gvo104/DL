"""
Дополнительные визуализации для отчёта.
Требуется: pip install pandas matplotlib seaborn
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Загрузка результатов
df = pd.read_csv("experiment_results.csv")

# Убедимся, что rubric числовой
df["rubric"] = pd.to_numeric(df["rubric"], errors="coerce")

# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)

# 1. Тепловая карта: rubric по темам и режимам
pivot = df.pivot_table(values="rubric", index="topic", columns="mode", aggfunc="mean")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=5, fmt=".1f")
plt.title("Средний rubric по темам и режимам")
plt.tight_layout()
plt.savefig("plots/heatmap_rubric.png", dpi=150)
plt.close()

# 2. Box plot: распределение rubric по режимам
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="mode", y="rubric", palette="Set2")
plt.title("Разброс rubric по режимам")
plt.ylabel("Rubric (0–5)")
plt.savefig("plots/boxplot_rubric.png", dpi=150)
plt.close()

# 3. Scatter plot: Latency vs Rubric
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="latency", y="rubric", hue="mode", style="mode", s=80)
plt.title("Задержка vs качество (rubric)")
plt.xlabel("Latency (сек)")
plt.ylabel("Rubric")
plt.legend(title="Режим")
plt.tight_layout()
plt.savefig("plots/latency_vs_rubric.png", dpi=150)
plt.close()

# 4. Корреляционная матрица числовых метрик
num_cols = ["rubric", "latency", "n_steps", "correctness", "groundedness", 
            "completeness", "coverage_of_required_fields", "source_consistency"]
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Корреляция метрик")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()

# 5. Сравнение per_page для agent (линейный график средних rubric)
agent_df = df[df["mode"] == "agent"].copy()
per_page_stats = agent_df.groupby("per_page").agg(
    mean_rubric=("rubric", "mean"),
    std_rubric=("rubric", "std"),
    mean_latency=("latency", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(per_page_stats["per_page"], per_page_stats["mean_rubric"], 
             yerr=per_page_stats["std_rubric"], marker='o', linewidth=2, 
             color='#2ca02c', label='Rubric')
ax1.set_xlabel("Количество источников (per_page)")
ax1.set_ylabel("Средний rubric", color='#2ca02c')
ax1.tick_params(axis='y', labelcolor='#2ca02c')

ax2 = ax1.twinx()
ax2.plot(per_page_stats["per_page"], per_page_stats["mean_latency"], 
         marker='s', linestyle='--', color='#d62728', label='Latency')
ax2.set_ylabel("Средняя задержка (сек)", color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title("Влияние количества источников на agent")
plt.tight_layout()
plt.savefig("plots/per_page_detail.png", dpi=150)
plt.close()

# 6. Влияние max_steps на agent
max_steps_df = agent_df[agent_df["per_page"] == 5]  # фиксируем per_page=5
max_steps_stats = max_steps_df.groupby("max_steps").agg(
    mean_rubric=("rubric", "mean"),
    std_rubric=("rubric", "std"),
    mean_latency=("latency", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(max_steps_stats["max_steps"], max_steps_stats["mean_rubric"], 
             yerr=max_steps_stats["std_rubric"], marker='o', linewidth=2,
             color='#1f77b4', label='Rubric')
ax1.set_xlabel("Максимальное число шагов (max_steps)")
ax1.set_ylabel("Средний rubric", color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = ax1.twinx()
ax2.plot(max_steps_stats["max_steps"], max_steps_stats["mean_latency"], 
         marker='s', linestyle='--', color='#ff7f0e', label='Latency')
ax2.set_ylabel("Средняя задержка (сек)", color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title("Влияние ограничения шагов на agent (per_page=5)")
plt.tight_layout()
plt.savefig("plots/max_steps_detail.png", dpi=150)
plt.close()

# 7. Сводная таблица средних значений по режимам
summary = df.groupby("mode").agg(
    mean_rubric=("rubric", "mean"),
    median_rubric=("rubric", "median"),
    std_rubric=("rubric", "std"),
    mean_latency=("latency", "mean"),
    mean_steps=("n_steps", "mean"),
    mean_groundedness=("groundedness", "mean"),
    mean_completeness=("completeness", "mean"),
    mean_source_consistency=("source_consistency", "mean")
).round(2)

print("\n=== Сводная таблица по режимам ===")
print(summary)
summary.to_csv("plots/summary_table.csv")

print("\nВсе графики сохранены в папку 'plots/'.")
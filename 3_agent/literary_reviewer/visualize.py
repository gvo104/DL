import pandas as pd
import matplotlib.pyplot as plt
import os

# Загрузка результатов
df = pd.read_csv("experiment_results.csv")

# Убедимся, что rubric числовой
df["rubric"] = pd.to_numeric(df["rubric"], errors="coerce")

# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)

# 1. Сравнение режимов (baseline, agent, agent_evaluator) по rubric и latency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
modes = ["baseline", "agent", "agent_evaluator"]
rubric_means = [df[df["mode"] == m]["rubric"].mean() for m in modes]
latency_means = [df[df["mode"] == m]["latency"].mean() for m in modes]

axes[0].bar(modes, rubric_means, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[0].set_title("Средний rubric по режимам")
axes[0].set_ylabel("Rubric (0-5)")

axes[1].bar(modes, latency_means, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[1].set_title("Средняя задержка (сек)")
axes[1].set_ylabel("Latency (s)")

plt.tight_layout()
plt.savefig("plots/mode_comparison.png")
plt.close()

# 2. Влияние количества источников (per_page) на agent
agent_df = df[df["mode"] == "agent"]
per_page_groups = agent_df.groupby("per_page")["rubric"].mean().reset_index()
plt.figure(figsize=(8, 5))
plt.bar(per_page_groups["per_page"].astype(str), per_page_groups["rubric"], color="skyblue")
plt.title("Agent: rubric vs количество источников (top-N)")
plt.xlabel("Количество источников")
plt.ylabel("Средний rubric")
plt.savefig("plots/per_page_impact.png")
plt.close()

# 3. Влияние max_steps на agent
max_steps_groups = agent_df.groupby("max_steps")["rubric"].mean().reset_index()
plt.figure(figsize=(8, 5))
plt.bar(max_steps_groups["max_steps"].astype(str), max_steps_groups["rubric"], color="salmon")
plt.title("Agent: rubric vs max_steps")
plt.xlabel("Max steps")
plt.ylabel("Средний rubric")
plt.savefig("plots/max_steps_impact.png")
plt.close()

# 4. Детализация по темам (heatmap или grouped bar)
pivot = df.pivot_table(values="rubric", index="topic", columns="mode", aggfunc="mean")
pivot.plot(kind="bar", figsize=(12, 6))
plt.title("Rubric по темам и режимам")
plt.ylabel("Rubric")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/rubric_by_topic.png")
plt.close()

print("Графики сохранены в папку 'plots'.")
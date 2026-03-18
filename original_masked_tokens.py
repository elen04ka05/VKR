import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Загрузка данных
df = pd.read_csv('snp_batch_processing_results_20260130_170214/tokenized_sequences.csv')

# Инициализация структур
num_sequences = len(df)
num_positions = 1107

# Словари для хранения частот: position -> {token: count}
before_masking_counts = {pos: Counter() for pos in range(num_positions)}
after_masking_counts = {pos: Counter() for pos in range(num_positions)}

# Разбор данных
for idx, row in df.iterrows():
    original_ids = list(map(int, row['original_ids'].split(';')))
    masked_ids = list(map(int, row['masked_ids'].split(';')))

    for pos, (orig_token, masked_token) in enumerate(zip(original_ids, masked_ids)):
        before_masking_counts[pos][orig_token] += 1
        if masked_token != 12109:  # Игнорируем маскированный токен
            after_masking_counts[pos][masked_token] += 1

# Теперь построим графики
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

# График до маскировки
ax1 = axes[0]
pos_list = list(range(1, num_positions + 1))
for pos in range(num_positions):
    tokens = list(before_masking_counts[pos].keys())
    counts = list(before_masking_counts[pos].values())
    if tokens:
        # Сортируем по токенам для единообразия
        sorted_tokens = sorted(tokens)
        sorted_counts = [before_masking_counts[pos][t] for t in sorted_tokens]
        # Рисуем столбики
        bottom = 0
        for token, count in zip(sorted_tokens, sorted_counts):
            ax1.bar(pos + 1, count, bottom=bottom, width=0.8, color=plt.cm.tab20(token % 20))
            bottom += count

ax1.set_title('Token Distribution Before Masking')
ax1.set_ylabel('Count')
ax1.set_xlim(0, num_positions + 1)

# График после маскировки
ax2 = axes[1]
for pos in range(num_positions):
    tokens = list(after_masking_counts[pos].keys())
    counts = list(after_masking_counts[pos].values())
    if tokens:
        sorted_tokens = sorted(tokens)
        sorted_counts = [after_masking_counts[pos][t] for t in sorted_tokens]
        bottom = 0
        for token, count in zip(sorted_tokens, sorted_counts):
            ax2.bar(pos + 1, count, bottom=bottom, width=0.8, color=plt.cm.tab20(token % 20))
            bottom += count

ax2.set_title('Token Distribution After Masking')
ax2.set_ylabel('Count')
ax2.set_xlabel('Token Position (1–1107)')

plt.tight_layout()
plt.show()
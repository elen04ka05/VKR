#токенизация SNP последовательностей с использованием k-mer подхода и случайного маскирования

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import json
import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt

file = "final_snp_matrix.csv"


class SNP_to_signal_k_mer:
    def __init__(self, k: int = 6, mask_prob: float = 0.15):
        self.k = k
        self.mask_prob = mask_prob
        self.vocab = {}
        self.mask_id = None

    def _create_kmers(self, sequence: List[str]) -> List[str]:#функция, которая разбивает последовательность на токены длиной k
        tokens = []
        for i in range(0, len(sequence), self.k):
            kmer = ''.join(sequence[i:i + self.k])
            if len(kmer) < self.k:
                kmer += 'X' * (self.k - len(kmer))
            tokens.append(kmer)
        return tokens

    def _build_vocab(self, all_kmers: List[str]): # создание словаря
        unique_kmers = set(all_kmers)
        self.vocab = {kmer: idx for idx, kmer in enumerate(unique_kmers)}
        self.mask_id = len(self.vocab)

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]: # берет список токенов и возвращает список их ID
        return [self.vocab[token] for token in tokens]

    def _apply_random_masking(self, token_ids: List[int]) -> List[int]: # маскирование
        masked_ids = []
        for token_id in token_ids:
            if random.random() < self.mask_prob:
                masked_ids.append(self.mask_id)
            else:
                masked_ids.append(token_id)
        return masked_ids

    def fit(self, snp_sequences: List[List[int]]): # создает словарь на основе всех последовательностей
        all_kmers = []

        for snp_seq in snp_sequences:
            kmers = self._create_kmers(snp_seq)
            all_kmers.extend(kmers)

        self._build_vocab(all_kmers)

        print(f"Словарь создан. Размер: {len(self.vocab)} токенов")
        print(f"Mask ID: {self.mask_id}")

    def transform(self, snp_sequence: List[int]) -> Tuple[List[int], List[str]]: # применяет все шаги к одной последовательности
        tokens = self._create_kmers(snp_sequence)
        token_ids = self._tokens_to_ids(tokens)
        masked_ids = self._apply_random_masking(token_ids)

        return masked_ids, tokens, token_ids  # Возвращаем также оригинальные ID

    def get_vocab_size(self) -> int: # получить  размер словаря
        return len(self.vocab) + 1


def batch_process_snp_sequences(snp_sequences: List[List[int]], k: int = 6, mask_prob: float = 0.15): # пакетная обработка
    processor = SNP_to_signal_k_mer(k=k, mask_prob=mask_prob)
    processor.fit(snp_sequences)

    results = []
    for i, seq in enumerate(snp_sequences):
        masked_ids, tokens, original_ids = processor.transform(seq)
        results.append({
            'sequence_id': i,
            'masked_ids': masked_ids,
            'original_ids': original_ids,
            'tokens': tokens,
            'original_length': len(seq),
            'compressed_length': len(tokens),
            'compression_ratio': len(seq) / len(tokens),
            'masked_count': sum(1 for x in masked_ids if x == processor.mask_id),
            'masked_percentage': sum(1 for x in masked_ids if x == processor.mask_id) / len(masked_ids) * 100
        })

    return processor, results


def load_and_filter_data(file_path): # читает файл и готовит данные к обработке
    df = pd.read_csv(file_path, sep=',', engine='python')

    print(f"Исходные данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

    nan_count = df.isna().sum().sum()
    print(f"Общее количество NaN значений в данных: {nan_count}")

    if nan_count > 0:
        print("Обработка NaN значений...")
        df = df.fillna('X')
        print("NaN значения заменены на 'X'")

    snp_data = df.iloc[:, 1:].values.tolist()

    snp_sequences = []
    for row in snp_data:
        sequence = [str(x) for x in row]
        snp_sequences.append(sequence)

    print(f"Загружено {len(snp_sequences)} образцов")
    print(f"Длина первого образца: {len(snp_sequences[0])} SNP")
    print(f"Пример первых 10 SNP первого образца: {snp_sequences[0][:10]}")

    return snp_sequences


def analyze_batch_results(processor: SNP_to_signal_k_mer, results: List[Dict]): # выводит общую статистику по обработке
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ПАКЕТНОЙ ОБРАБОТКИ")
    print("=" * 60)

    total_sequences = len(results)
    avg_original_length = np.mean([r['original_length'] for r in results])
    avg_compressed_length = np.mean([r['compressed_length'] for r in results])
    avg_compression_ratio = np.mean([r['compression_ratio'] for r in results])
    avg_masked_percentage = np.mean([r['masked_percentage'] for r in results])

    print(f"Обработано последовательностей: {total_sequences}")
    print(f"Размер словаря: {processor.get_vocab_size()} токенов")
    print(f"Средняя оригинальная длина: {avg_original_length:.0f} SNP")
    print(f"Средняя длина после k-mer: {avg_compressed_length:.0f} токенов")
    print(f"Средний коэффициент сжатия: {avg_compression_ratio:.2f}x")
    print(f"Средний процент маскирования: {avg_masked_percentage:.1f}%")
    print(f"Целевой процент маскирования: {processor.mask_prob * 100:.1f}%")


def print_detailed_examples(results: List[Dict], processor: SNP_to_signal_k_mer, num_examples: int = 3): #показывает подробно несколько примеров
    print(f"\nДЕТАЛИ ОБРАБОТКИ (первые {num_examples} последовательностей):")
    print("=" * 60)

    for i in range(min(num_examples, len(results))):
        result = results[i]
        print(f"\n--- Последовательность #{result['sequence_id']} ---")
        print(f"Оригинальная длина: {result['original_length']} SNP")
        print(f"После k-mer: {result['compressed_length']} токенов")
        print(f"Коэффициент сжатия: {result['compression_ratio']:.2f}x")
        print(f"Замаскировано токенов: {result['masked_count']} ({result['masked_percentage']:.1f}%)")

        print("\nПервые 5 токенов:")
        for j, (token, original_id, masked_id) in enumerate(zip(result['tokens'][:5],
                                                                result['original_ids'][:5],
                                                                result['masked_ids'][:5])):
            mask_status = "[MASK]" if masked_id == processor.mask_id else ""
            print(f"  {j}: '{token}' -> Original ID: {original_id} | Masked ID: {masked_id} {mask_status}")


def save_token_id_comparison(results: List[Dict], results_dir: str): # сохраняет в CSV файл все ID до и после маскировки
    """
    Сохраняет сравнение ID токенов до и после маскировки
    """
    filename = os.path.join(results_dir, "token_id_comparison.csv")

    all_data = []
    for result in results:
        row = {
            'sequence_id': result['sequence_id'],
            'original_ids': ';'.join(map(str, result['original_ids'])),
            'masked_ids': ';'.join(map(str, result['masked_ids'])),
            'tokens': ';'.join(result['tokens']),
            'masked_count': result['masked_count']
        }
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False, encoding='utf-8')

    print(f"Файл сравнения ID токенов сохранен: {filename}")
    return filename


def calculate_token_statistics(processor: SNP_to_signal_k_mer, results: List[Dict]): #считает, как часто встречается каждый токен
    """
    Рассчитывает статистику токенов до и после маскировки
    """
    # Собираем все оригинальные и замаскированные ID
    all_original_ids = []
    all_masked_ids = []

    for result in results:
        all_original_ids.extend(result['original_ids'])
        all_masked_ids.extend(result['masked_ids'])

    # Считаем частоты
    original_counts = Counter(all_original_ids)
    masked_counts = Counter(all_masked_ids)

    # Создаем DataFrame с частотой каждого токена
    token_stats = []
    for token_id in sorted(original_counts.keys()):
        # Находим токен по ID (обратный поиск в словаре)
        token = None
        for kmer, t_id in processor.vocab.items():
            if t_id == token_id:
                token = kmer
                break
        if token is None:
            token = "UNKNOWN"

        token_stats.append({
            'token_id': token_id,
            'token': token,
            'original_frequency': original_counts[token_id],
            'masked_frequency': masked_counts[token_id],
            'masked_as_mask': masked_counts.get(processor.mask_id, 0),
            'difference': masked_counts[token_id] - original_counts[token_id],
            'is_masked_token': token_id == processor.mask_id
        })

    # Добавляем статистику для mask токена отдельно
    mask_stats = {
        'token_id': processor.mask_id,
        'token': '[MASK]',
        'original_frequency': 0,
        'masked_frequency': masked_counts.get(processor.mask_id, 0),
        'masked_as_mask': masked_counts.get(processor.mask_id, 0),
        'difference': masked_counts.get(processor.mask_id, 0),
        'is_masked_token': True
    }
    token_stats.append(mask_stats)

    return pd.DataFrame(token_stats), original_counts, masked_counts


def save_token_statistics(token_stats_df: pd.DataFrame, results_dir: str):
    """
    Сохраняет статистику токенов в файл
    """
    filename = os.path.join(results_dir, "token_statistics.csv")
    token_stats_df.to_csv(filename, index=False, encoding='utf-8')

    # Также сохраняем сводную статистику
    summary_filename = os.path.join(results_dir, "token_statistics_summary.txt")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("СВОДНАЯ СТАТИСТИКА ТОКЕНОВ\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Всего уникальных токенов: {len(token_stats_df)}\n")
        f.write(f"Общее количество токенов до маскировки: {token_stats_df['original_frequency'].sum()}\n")
        f.write(f"Общее количество токенов после маскировки: {token_stats_df['masked_frequency'].sum()}\n")
        f.write(
            f"Количество [MASK] токенов: {token_stats_df[token_stats_df['token'] == '[MASK]']['masked_frequency'].values[0]}\n\n")

        # Топ-10 самых частых токенов до маскировки
        f.write("ТОП-10 самых частых токенов до маскировки:\n")
        top_original = token_stats_df[token_stats_df['token'] != '[MASK]'].nlargest(10, 'original_frequency')
        for idx, row in top_original.iterrows():
            f.write(f"  {row['token']} (ID: {row['token_id']}): {row['original_frequency']} раз\n")

        f.write("\nТОП-10 самых частых токенов после маскировки:\n")
        top_masked = token_stats_df[token_stats_df['token'] != '[MASK]'].nlargest(10, 'masked_frequency')
        for idx, row in top_masked.iterrows():
            f.write(f"  {row['token']} (ID: {row['token_id']}): {row['masked_frequency']} раз\n")

        # Токены с наибольшим изменением частоты
        f.write("\nТокены с наибольшим уменьшением частоты (заменены на [MASK]):\n")
        top_decreased = token_stats_df[token_stats_df['token'] != '[MASK]'].nsmallest(10, 'difference')
        for idx, row in top_decreased.iterrows():
            f.write(
                f"  {row['token']} (ID: {row['token_id']}): было {row['original_frequency']}, стало {row['masked_frequency']} (изменение: {row['difference']})\n")

    print(f"Статистика токенов сохранена: {filename}")
    print(f"Сводная статистика сохранена: {summary_filename}")

    return filename, summary_filename


def create_token_frequency_by_samples_plots(processor: SNP_to_signal_k_mer, results: List[Dict], results_dir: str,
                                            top_n_tokens: int = None, max_samples_per_plot: int = None):
    """
    Создает графики частот токенов по образцам (до и после маскировки)
    Анализирует ВСЕ токены по ВСЕМ образцам

    Args:
        processor: Объект процессора с словарем
        results: Результаты обработки
        results_dir: Директория для сохранения результатов
        top_n_tokens: Если указано, анализирует только top_n_tokens самых частых токенов
        max_samples_per_plot: Если указано, ограничивает количество образцов на графике
    """
    print("\nСоздание графиков частот токенов по образцам...")
    print(f"Всего токенов в словаре: {len(processor.vocab)}")
    print(f"Всего образцов: {len(results)}")

    # Если параметры не указаны, используем все данные
    if top_n_tokens is None:
        top_n_tokens = len(processor.vocab)
    if max_samples_per_plot is None:
        max_samples_per_plot = len(results)

    print(f"Анализируем все {top_n_tokens} токенов...")
    print(f"Анализируем все {max_samples_per_plot} образцов...")

    # Создаем папку для графиков по токенам
    plots_dir = os.path.join(results_dir, "token_frequency_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Определяем самые частые токены до маскировки
    token_freq_data = []

    # Создаем обратный словарь для быстрого поиска токена по ID
    reverse_vocab = {v: k for k, v in processor.vocab.items()}

    # Сначала собираем частоты всех токенов
    for token_id in range(len(processor.vocab)):
        total_freq = 0
        for result in results:
            total_freq += result['original_ids'].count(token_id)

        token = reverse_vocab.get(token_id, f"TOKEN_{token_id}")
        token_freq_data.append({
            'token': token,
            'token_id': token_id,
            'total_frequency': total_freq
        })

    # Сортируем по частоте
    token_freq_df = pd.DataFrame(token_freq_data)
    token_freq_df = token_freq_df.sort_values('total_frequency', ascending=False)

    # Берем указанное количество токенов или все
    if top_n_tokens < len(token_freq_df):
        top_tokens = token_freq_df.head(top_n_tokens)
        print(f"Анализируем {top_n_tokens} самых частых токенов из {len(token_freq_df)}...")
    else:
        top_tokens = token_freq_df
        print(f"Анализируем ВСЕ {len(top_tokens)} токенов...")

    # Для каждого токена создаем график
    created_plots = []
    zero_frequency_tokens = 0

    for idx, token_row in top_tokens.iterrows():
        token = token_row['token']
        token_id = token_row['token_id']
        total_freq = token_row['total_frequency']

        # Собираем данные для этого токена по всем образцам
        sample_ids = []
        original_freqs = []
        masked_freqs = []

        for result in results[:max_samples_per_plot]:  # Используем все образцы
            sample_id = result['sequence_id']

            # Частота до маскировки
            original_freq = result['original_ids'].count(token_id)

            # Частота после маскировки (без учета случаев, когда токен замаскирован)
            # Если токен был замаскирован, он превратился в [MASK], поэтому не считается
            masked_freq = 0
            for orig_id, masked_id in zip(result['original_ids'], result['masked_ids']):
                if orig_id == token_id and masked_id == token_id:  # Токен не был замаскирован
                    masked_freq += 1

            sample_ids.append(sample_id)
            original_freqs.append(original_freq)
            masked_freqs.append(masked_freq)

        # Проверяем, есть ли данные для этого токена
        if total_freq == 0:
            zero_frequency_tokens += 1
            continue

        # Создаем DataFrame для этого токена
        token_data = pd.DataFrame({
            'sample_id': sample_ids,
            'original_frequency': original_freqs,
            'masked_frequency': masked_freqs
        })

        # Создаем график
        plt.figure(figsize=(max(14, len(sample_ids) * 0.1), 6))  # Адаптируем ширину под количество образцов

        # Линия для частот до маскировки
        plt.plot(token_data['sample_id'], token_data['original_frequency'],
                 marker='o', linestyle='-', linewidth=1, markersize=3,
                 color='blue', alpha=0.7, label='До маскировки')

        # Линия для частот после маскировки
        plt.plot(token_data['sample_id'], token_data['masked_frequency'],
                 marker='s', linestyle='--', linewidth=1, markersize=3,
                 color='red', alpha=0.7, label='После маскировки')

        # Заполняем область между линиями (разница)
        plt.fill_between(token_data['sample_id'],
                         token_data['original_frequency'],
                         token_data['masked_frequency'],
                         alpha=0.2, color='gray', label='Разница')

        plt.xlabel('Номер образца', fontsize=12)
        plt.ylabel('Частота встречаемости', fontsize=12)
        plt.title(
            f'Частота токена "{token}" (ID: {token_id}) по образцам\nВсего вхождений: {total_freq}\nДо и после маскировки',
            fontsize=14, fontweight='bold')

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)

        # Настраиваем ось X для большого количества образцов
        if len(sample_ids) > 50:
            # Показываем не все метки на оси X
            step = max(1, len(sample_ids) // 20)
            plt.xticks(sample_ids[::step], fontsize=8, rotation=45)
        else:
            plt.xticks(fontsize=10)

        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Сохраняем график
        # Очищаем имя файла от специальных символов
        safe_token_name = ''.join(c if c.isalnum() else '_' for c in token)[:50]  # Ограничиваем длину имени
        plot_filename = os.path.join(plots_dir, f"token_{token_id:04d}_{safe_token_name}_frequency.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

        # Сохраняем данные в CSV
        data_filename = os.path.join(plots_dir, f"token_{token_id:04d}_{safe_token_name}_data.csv")
        token_data.to_csv(data_filename, index=False, encoding='utf-8')

        created_plots.append((plot_filename, data_filename))

        # Выводим прогресс каждые 50 токенов
        if len(created_plots) % 50 == 0:
            print(f"  Обработано {len(created_plots)} токенов...")

    print(f"\nОбработано токенов:")
    print(f"  - Всего токенов: {len(top_tokens)}")
    print(f"  - Создано графиков: {len(created_plots)}")
    print(f"  - Токены без вхождений: {zero_frequency_tokens}")

    if created_plots:
        # Сохраняем индекс всех токенов
        #save_token_index(top_tokens, created_plots, plots_dir)

        # Создаем сводный график для нескольких токенов
        #create_summary_frequency_plot(top_tokens, results, plots_dir, max_samples_per_plot)

        # Создаем сводный отчет
        create_frequency_summary_report(top_tokens, results, plots_dir, max_samples_per_plot)

        print(f"\nСоздано {len(created_plots)} графиков для отдельных токенов")
        print(f"Все графики сохранены в папке: {plots_dir}")
    else:
        print("Не удалось создать графики - нет токенов с вхождениями в выбранных образцах")

    return plots_dir


def save_token_index(top_tokens: pd.DataFrame, created_plots: List[Tuple], plots_dir: str):
    """
    Сохраняет индекс всех токенов с ссылками на их графики
    """
    index_filename = os.path.join(plots_dir, "token_index.html")

    with open(index_filename, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Индекс графиков токенов</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .token-cell { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    </style>
</head>
<body>
    <h1>Индекс графиков токенов</h1>
    <p>Всего токенов: {}</p>
    <table>
        <tr>
            <th>ID токена</th>
            <th>Токен</th>
            <th>Общая частота</th>
            <th>График</th>
            <th>Данные</th>
        </tr>
""".format(len(top_tokens)))

        # Сортируем по ID для удобства
        sorted_tokens = top_tokens.sort_values('token_id')

        for idx, token_row in sorted_tokens.iterrows():
            token = token_row['token']
            token_id = token_row['token_id']
            total_freq = token_row['total_frequency']

            # Находим соответствующие файлы
            plot_file = None
            data_file = None

            for plot_file_path, data_file_path in created_plots:
                if f"token_{token_id:04d}" in plot_file_path:
                    plot_file = os.path.basename(plot_file_path)
                    data_file = os.path.basename(data_file_path)
                    break

            if plot_file and data_file:
                f.write(f"""        <tr>
            <td>{token_id}</td>
            <td class="token-cell" title="{token}">{token}</td>
            <td>{total_freq}</td>
            <td><a href="{plot_file}">{plot_file}</a></td>
            <td><a href="{data_file}">{data_file}</a></td>
        </tr>
""")

        f.write("""    </table>
</body>
</html>""")

    print(f"HTML индекс создан: {index_filename}")


def create_summary_frequency_plot(top_tokens: pd.DataFrame, results: List[Dict],
                                  plots_dir: str, max_samples_per_plot: int = None):
    """
    Создает сводный график с несколькими токенами
    """
    if max_samples_per_plot is None:
        max_samples_per_plot = len(results)

    plt.figure(figsize=(16, 10))

    # Ограничиваем количество токенов для сводного графика (первые 20)
    plot_tokens = top_tokens.head(20)

    # Цветовая палитра для токенов
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_tokens)))

    # Для каждого токена добавляем данные
    for idx, (_, token_row) in enumerate(plot_tokens.iterrows()):
        token = token_row['token']
        token_id = token_row['token_id']

        # Собираем данные
        sample_ids = []
        original_freqs = []

        for result in results[:max_samples_per_plot]:
            sample_ids.append(result['sequence_id'])
            original_freqs.append(result['original_ids'].count(token_id))

        # Сглаживаем данные для лучшей визуализации
        if len(sample_ids) > 5:
            # Используем скользящее среднее
            window_size = min(5, len(sample_ids) // 10 + 1)
            smoothed_freqs = pd.Series(original_freqs).rolling(window=window_size, center=True).mean()
            # Заполняем NaN значения
            smoothed_freqs = smoothed_freqs.fillna(method='bfill').fillna(method='ffill')
        else:
            smoothed_freqs = original_freqs

        plt.plot(sample_ids, smoothed_freqs,
                 color=colors[idx], alpha=0.7, linewidth=2,
                 label=f'{token} (ID:{token_id})')

    plt.xlabel('Номер образца', fontsize=12)
    plt.ylabel('Частота встречаемости (до маскировки)', fontsize=12)
    plt.title('Сводный график частот токенов по образцам (топ-20 токенов)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Настраиваем ось X для большого количества образцов
    if max_samples_per_plot > 50:
        step = max(1, max_samples_per_plot // 20)
        plt.xticks(range(0, max_samples_per_plot, step), fontsize=8, rotation=45)
    else:
        plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Сохраняем сводный график
    summary_filename = os.path.join(plots_dir, "summary_token_frequencies.png")
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Сводный график сохранен: {summary_filename}")


def create_frequency_summary_report(top_tokens: pd.DataFrame, results: List[Dict],
                                    plots_dir: str, max_samples_per_plot: int = None): # Создает сводный отчет по частотам токенов
    """
    Создает сводный отчет по частотам токенов
    """
    if max_samples_per_plot is None:
        max_samples_per_plot = len(results)

    report_filename = os.path.join(plots_dir, "frequency_analysis_report.txt")

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("СВОДНЫЙ ОТЧЕТ ПО ЧАСТОТАМ ТОКЕНОВ ПО ОБРАЗЦАМ\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Проанализировано токенов: {len(top_tokens)}\n")
        f.write(f"Проанализировано образцов: {max_samples_per_plot}\n\n")

        # Общая статистика
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write("-" * 40 + "\n")

        total_tokens_before = sum(len(r['original_ids']) for r in results[:max_samples_per_plot])
        total_tokens_after = sum(len(r['masked_ids']) for r in results[:max_samples_per_plot])
        total_masked = sum(r['masked_count'] for r in results[:max_samples_per_plot])

        f.write(f"Всего токенов до маскировки: {total_tokens_before}\n")
        f.write(f"Всего токенов после маскировки: {total_tokens_after}\n")
        f.write(f"Всего замаскировано токенов: {total_masked}\n")
        f.write(f"Общий процент маскирования: {total_masked / total_tokens_before * 100:.1f}%\n\n")

        f.write("СТАТИСТИКА ПО ТОКЕНАМ (топ-50):\n")
        f.write("-" * 40 + "\n")

        # Берем топ-50 токенов для отчета
        top_50_tokens = top_tokens.head(50)

        for idx, token_row in top_50_tokens.iterrows():
            token = token_row['token']
            token_id = token_row['token_id']
            total_freq = token_row['total_frequency']

            # Собираем статистику
            total_masked_for_token = 0
            present_in_samples = 0

            for result in results[:max_samples_per_plot]:
                original_freq = result['original_ids'].count(token_id)
                if original_freq > 0:
                    present_in_samples += 1

                    # Считаем, сколько раз токен был замаскирован
                    for orig_id, masked_id in zip(result['original_ids'], result['masked_ids']):
                        if orig_id == token_id and masked_id != token_id:  # Токен был замаскирован
                            total_masked_for_token += 1

            avg_per_sample = total_freq / max(present_in_samples, 1) if present_in_samples > 0 else 0
            masking_rate = total_masked_for_token / max(total_freq, 1) if total_freq > 0 else 0

            f.write(f"\nТокен {idx + 1:2d}: '{token}' (ID: {token_id})\n")
            f.write(f"  Всего вхождений: {total_freq}\n")
            f.write(f"  Присутствует в {present_in_samples} образцах\n")
            f.write(f"  Среднее вхождений на образец: {avg_per_sample:.2f}\n")
            f.write(f"  Замаскировано: {total_masked_for_token} ({masking_rate * 100:.1f}%)\n")

    print(f"Сводный отчет сохранен: {report_filename}")


def plot_token_frequencies(token_stats_df: pd.DataFrame, results_dir: str, top_n: int = 30): #Создает графики частот токенов до и после маскировки
    """
    Создает графики частот токенов до и после маскировки
    """
    # Исключаем [MASK] токен для лучшей визуализации
    plot_df = token_stats_df[token_stats_df['token'] != '[MASK]'].copy()

    # Берем top_n самых частых токенов
    plot_df = plot_df.nlargest(top_n, 'original_frequency')

    # Создаем график
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # График 1: Частоты до и после маскировки
    x = np.arange(len(plot_df))
    width = 0.35

    axes[0].bar(x - width / 2, plot_df['original_frequency'], width, label='До маскировки', alpha=0.8)
    axes[0].bar(x + width / 2, plot_df['masked_frequency'], width, label='После маскировки', alpha=0.8)
    axes[0].set_xlabel('Токены')
    axes[0].set_ylabel('Частота')
    axes[0].set_title(f'Частота токенов до и после маскировки (топ-{top_n})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t}\n(ID:{id})" for t, id in zip(plot_df['token'], plot_df['token_id'])],
                            rotation=45, ha='right', fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График 2: Изменение частоты
    axes[1].bar(x, plot_df['difference'], alpha=0.7, color='red')
    axes[1].set_xlabel('Токены')
    axes[1].set_ylabel('Изменение частоты')
    axes[1].set_title(f'Изменение частоты токенов после маскировки (топ-{top_n})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t}\n(ID:{id})" for t, id in zip(plot_df['token'], plot_df['token_id'])],
                            rotation=45, ha='right', fontsize=8)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохраняем график
    plot_filename = os.path.join(results_dir, "token_frequency_comparison.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Создаем график распределения частот
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Определяем общий диапазон для обеих гистограмм
    all_values = pd.concat([plot_df['original_frequency'], plot_df['masked_frequency']])
    min_val = all_values.min()
    max_val = all_values.max()

    # Создаем одинаковые бины для обоих графиков
    bins = np.linspace(min_val, max_val, 21)  # 20 интервалов

    # Гистограмма частот до маскировки
    axes[0].hist(plot_df['original_frequency'], bins=bins, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Частота')
    axes[0].set_ylabel('Количество токенов')
    axes[0].set_title('Распределение частот до маскировки')
    axes[0].set_xlim(min_val, max_val)  # Фиксируем одинаковые пределы по X
    axes[0].grid(True, alpha=0.3)

    # Гистограмма частот после маскировки
    axes[1].hist(plot_df['masked_frequency'], bins=bins, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Частота')
    axes[1].set_ylabel('Количество токенов')
    axes[1].set_title('Распределение частот после маскировки')
    axes[1].set_xlim(min_val, max_val)  # Фиксируем одинаковые пределы по X
    axes[1].grid(True, alpha=0.3)

    # Для еще лучшего сравнения можно также установить одинаковые пределы по Y
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, y_max)
    axes[1].set_ylim(0, y_max)

    plt.tight_layout()

    hist_filename = os.path.join(results_dir, "token_frequency_distributions.png")
    plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Графики сохранены:")
    print(f"  - {plot_filename}")
    print(f"  - {hist_filename}")

    return plot_filename, hist_filename


def save_results_to_files(processor: SNP_to_signal_k_mer, results: List[Dict], base_filename: str = "snp_processing"):
    """
    Сохранение результатов обработки в различные файлы
    """
    import os
    import time

    # Создаем папку для результатов
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_filename}_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nСохранение результатов в папку: {results_dir}")

    # 1. Сохранение токенизированных данных в CSV
    csv_filename = os.path.join(results_dir, "tokenized_sequences.csv")
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write(
            "sequence_id,original_length,compressed_length,compression_ratio,masked_count,masked_percentage,original_ids,masked_ids,tokens\n")
        for result in results:
            # Преобразуем списки в строки для CSV
            original_ids_str = ';'.join(map(str, result['original_ids']))
            masked_ids_str = ';'.join(map(str, result['masked_ids']))
            tokens_str = ';'.join(result['tokens'])
            f.write(
                f"{result['sequence_id']},{result['original_length']},{result['compressed_length']},{result['compression_ratio']:.2f},{result['masked_count']},{result['masked_percentage']:.1f},\"{original_ids_str}\",\"{masked_ids_str}\",\"{tokens_str}\"\n")

    # 2. Сохранение словаря в JSON
    vocab_filename = os.path.join(results_dir, "vocabulary.json")
    with open(vocab_filename, 'w', encoding='utf-8') as f:
        json.dump(processor.vocab, f, indent=2, ensure_ascii=False)

    # 3. Сохранение статистики в TXT
    stats_filename = os.path.join(results_dir, "processing_statistics.txt")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА ОБРАБОТКИ SNP ПОСЛЕДОВАТЕЛЬНОСТЕЙ\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Параметры обработки:\n")
        f.write(f"  k-mer размер: {processor.k}\n")
        f.write(f"  Вероятность маскирования: {processor.mask_prob}\n")
        f.write(f"  Размер словаря: {processor.get_vocab_size()}\n")
        f.write(f"  Mask ID: {processor.mask_id}\n\n")

        # Общая статистика
        total_sequences = len(results)
        avg_original_length = np.mean([r['original_length'] for r in results])
        avg_compressed_length = np.mean([r['compressed_length'] for r in results])
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in results])
        avg_masked_percentage = np.mean([r['masked_percentage'] for r in results])

        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write(f"  Обработано последовательностей: {total_sequences}\n")
        f.write(f"  Средняя оригинальная длина: {avg_original_length:.0f} SNP\n")
        f.write(f"  Средняя длина после k-mer: {avg_compressed_length:.0f} токенов\n")
        f.write(f"  Средний коэффициент сжатия: {avg_compression_ratio:.2f}x\n")
        f.write(f"  Средний процент маскирования: {avg_masked_percentage:.1f}%\n\n")

        # Детальная статистика по последовательностям
        f.write("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ:\n")
        f.write("-" * 50 + "\n")
        for result in results:
            f.write(f"Последовательность {result['sequence_id']}:\n")
            f.write(f"  Оригинальная длина: {result['original_length']} SNP\n")
            f.write(f"  Сжатая длина: {result['compressed_length']} токенов\n")
            f.write(f"  Коэффициент сжатия: {result['compression_ratio']:.2f}x\n")
            f.write(f"  Замаскировано: {result['masked_count']} токенов ({result['masked_percentage']:.1f}%)\n")
            f.write(f"  Первые 3 токена: {result['tokens'][:3]}\n\n")

    # 4. Сохранение бинарных данных (для дальнейшей обработки)
    binary_filename = os.path.join(results_dir, "processed_data.pkl")
    with open(binary_filename, 'wb') as f:
        pickle.dump({
            'processor': processor,
            'results': results,
            'parameters': {
                'k': processor.k,
                'mask_prob': processor.mask_prob,
                'vocab_size': processor.get_vocab_size()
            }
        }, f)

    # 5. Сохранение примеров токенов
    examples_filename = os.path.join(results_dir, "token_examples.txt")
    with open(examples_filename, 'w', encoding='utf-8') as f:
        f.write("ПРИМЕРЫ ТОКЕНОВ И ИХ ID\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Всего уникальных k-mer: {len(processor.vocab)}\n")
        f.write(f"Mask ID: {processor.mask_id}\n\n")

        f.write("Первые 20 k-mer из словаря:\n")
        for i, (kmer, token_id) in enumerate(list(processor.vocab.items())[:20]):
            f.write(f"  {i + 1:2d}. '{kmer}' -> ID: {token_id}\n")

        f.write(f"\nПримеры из последовательностей:\n")
        for i in range(min(3, len(results))):
            result = results[i]
            f.write(f"\nПоследовательность #{result['sequence_id']}:\n")
            f.write(f"Первые 10 токенов:\n")
            for j, (token, original_id, masked_id) in enumerate(zip(result['tokens'][:10],
                                                                    result['original_ids'][:10],
                                                                    result['masked_ids'][:10])):
                mask_status = "[MASK]" if masked_id == processor.mask_id else ""
                f.write(f"  {j}: '{token}' -> Original ID: {original_id} | Masked ID: {masked_id} {mask_status}\n")

    # 6. Сохраняем сравнение ID токенов
    token_comparison_file = save_token_id_comparison(results, results_dir)

    # 7. Рассчитываем и сохраняем статистику токенов
    token_stats_df, original_counts, masked_counts = calculate_token_statistics(processor, results)
    token_stats_file, token_summary_file = save_token_statistics(token_stats_df, results_dir)

    # 8. Создаем графики частот токенов
    plot_files = plot_token_frequencies(token_stats_df, results_dir, top_n=30)

    # 9. Создаем графики частот по образцам для каждого токена
    try:
        frequency_plots_dir = create_token_frequency_by_samples_plots(
            processor, results, results_dir,
            top_n_tokens=None,  # Все токены
            max_samples_per_plot=None  # Все образцы
        )
        print(f"  - Графики по образцам сохранены в: {frequency_plots_dir}")
    except Exception as e:
        print(f"  - Ошибка при создании графиков по образцам: {e}")
        frequency_plots_dir = None

    # 10. Создаем CSV файл только с ID последовательностей после маскирования
    masked_csv_file = save_masked_ids_csv(processor, results, results_dir, "masked_sequences_ids.csv")
    print(f"  - {masked_csv_file} (CSV с ID после маскирования)")

    print(f"\nФайлы сохранены:")
    print(f"  - {csv_filename} (токенизированные данные)")
    print(f"  - {vocab_filename} (словарь)")
    print(f"  - {stats_filename} (статистика)")
    print(f"  - {binary_filename} (бинарные данные)")
    print(f"  - {examples_filename} (примеры токенов)")
    print(f"  - {token_comparison_file} (сравнение ID токенов)")
    print(f"  - {token_stats_file} (статистика токенов)")
    print(f"  - {token_summary_file} (сводная статистика)")
    print(f"  - {plot_files[0]} (график сравнения частот)")
    print(f"  - {plot_files[1]} (график распределения частот)")

    return results_dir


def save_masked_ids_csv(processor: SNP_to_signal_k_mer, results: List[Dict], results_dir: str,
                        filename: str = "masked_sequences_ids.csv"):
    """
    СОЗДАЕТ CSV ФАЙЛ С ID ПОСЛЕДОВАТЕЛЬНОСТЕЙ ПОСЛЕ МАСКИРОВАНИЯ

    Формат выходного файла:
    - Каждая строка = один образец
    - В строке только ID через запятую (без заголовков, без индексов)
    - ID - это числа после применения маскирования (некоторые заменены на mask_id)

    Пример строки:
    12109,45,12,78,12109,3,67,12109,89,12,...

    Args:
        processor: процессор с данными
        results: результаты обработки
        results_dir: папка для сохранения
        filename: имя CSV файла

    Returns:
        путь к созданному файлу
    """

    # Создаем полный путь к файлу
    filepath = os.path.join(results_dir, filename)

    print(f"\nСоздаю CSV файл с ID последовательностей (после маскирования): {filepath}")

    # Открываем файл для записи
    with open(filepath, 'w', encoding='utf-8') as f:
        # Проходим по всем результатам (образцам)
        for result in results:
            # Берем masked_ids - это ID после маскирования
            masked_ids = result['masked_ids']

            # Преобразуем каждый ID в строку и объединяем через запятую
            # Пример: [12109, 45, 12, 12109, 78] -> "12109,45,12,12109,78"
            ids_string = ','.join(str(id) for id in masked_ids)

            # Записываем строку в файл
            f.write(ids_string + '\n')

    # Считаем статистику
    num_sequences = len(results)

    # Берем первый образец для примера
    first_sample_ids = results[0]['masked_ids']
    first_sample_example = ','.join(str(id) for id in first_sample_ids[:20]) + '...'

    # Считаем количество маскированных токенов
    total_masked = sum(result['masked_count'] for result in results)
    total_tokens = sum(len(result['masked_ids']) for result in results)
    avg_masked_percent = (total_masked / total_tokens * 100) if total_tokens > 0 else 0

    print(f"✅ CSV файл создан!")
    print(f"   - Сохранено образцов: {num_sequences}")
    print(f"   - Длина каждого образца: {len(first_sample_ids)} токенов")
    print(f"   - Всего токенов в файле: {total_tokens}")
    print(f"   - Из них маскированных: {total_masked} ({avg_masked_percent:.1f}%)")
    print(f"   - ID маски: {processor.mask_id}")
    print(f"   - Пример первой строки: {first_sample_example}")
    print(f"   - Полный путь: {filepath}")

    # Создаем информационный файл с описанием формата
    info_filepath = os.path.join(results_dir, "masked_sequences_README.txt")
    with open(info_filepath, 'w', encoding='utf-8') as info_f:
        info_f.write("ОПИСАНИЕ ФАЙЛА masked_sequences_ids.csv\n")
        info_f.write("=" * 50 + "\n\n")
        info_f.write("ФОРМАТ ФАЙЛА:\n")
        info_f.write("- CSV файл без заголовков\n")
        info_f.write("- Каждая строка = один образец (SNP последовательность)\n")
        info_f.write("- Числа в строке = ID токенов ПОСЛЕ МАСКИРОВАНИЯ\n")
        info_f.write("- ID разделены запятыми\n")
        info_f.write("- НЕТ пробелов, только числа и запятые\n\n")

        info_f.write("ПРИМЕР СТРОКИ:\n")
        info_f.write(f"{','.join(str(id) for id in first_sample_ids[:10])}\n\n")

        info_f.write("СТАТИСТИКА:\n")
        info_f.write(f"  Всего образцов: {num_sequences}\n")
        info_f.write(f"  Длина каждого образца: {len(first_sample_ids)} токенов\n")
        info_f.write(f"  Всего токенов: {total_tokens}\n")
        info_f.write(f"  Маскированных токенов: {total_masked} ({avg_masked_percent:.1f}%)\n")
        info_f.write(f"  ID маски: {processor.mask_id}\n\n")

        info_f.write("ПАРАМЕТРЫ ОБРАБОТКИ:\n")
        info_f.write(f"  k-mer размер: {processor.k}\n")
        info_f.write(f"  Вероятность маскирования: {processor.mask_prob}\n")
        info_f.write(f"  Размер словаря: {processor.get_vocab_size()}\n")
        info_f.write(f"  Всего уникальных токенов (без маски): {len(processor.vocab)}\n\n")

        info_f.write("КАК ИСПОЛЬЗОВАТЬ:\n")
        info_f.write("Этот файл готов для подачи в нейросеть.\n")
        info_f.write("Каждая строка может быть загружена как:\n")
        info_f.write("  import pandas as pd\n")
        info_f.write("  df = pd.read_csv('masked_sequences_ids.csv', header=None)\n")
        info_f.write("  # или как список списков:\n")
        info_f.write("  with open('masked_sequences_ids.csv', 'r') as f:\n")
        info_f.write("      sequences = [list(map(int, line.strip().split(','))) for line in f]\n")

    print(f"   - Информационный файл: {info_filepath}")

    return filepath

if __name__ == "__main__":
    print("ПАКЕТНАЯ ОБРАБОТКА SNP ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
    print("=" * 60)

    snp_sequences = load_and_filter_data(file)

    # Пакетная обработка с параметрами из статьи
    print("\nЗапуск пакетной обработки...")
    processor, results = batch_process_snp_sequences(
        snp_sequences=snp_sequences,
        k=6,  # 6-mer как в статье
        mask_prob=0.15  # 15% маскирование как в статье
    )

    # Анализ результатов
    analyze_batch_results(processor, results)

    for i, (kmer, token_id) in enumerate(list(processor.vocab.items())[:20]):
        print(f"  '{kmer}' -> {token_id}")

    results_dir = save_results_to_files(processor, results, "snp_batch_processing")
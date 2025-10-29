import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import json
import pickle

file = "total_df_for_aio_chickpea_28042016_synchro.csv"


class SNP_to_signal_k_mer:
    def __init__(self, k: int = 6, mask_prob: float = 0.15):
        self.k = k
        self.mask_prob = mask_prob
        self.vocab = {}
        self.mask_id = None
        self.letter_map = {
            0: 'A',  # AA
            1: 'R',  # Aa
            2: 'T',  # aa
        }

    def _snp_to_letters(self, snp_sequence: List[int]) -> List[str]:
        return [self.letter_map.get(s, 'N') for s in snp_sequence]

    def _preprocess_sequence(self, letter_sequence: List[str]) -> List[str]:
        return [s if s in {'A', 'T', 'C', 'G'} else 'X' for s in letter_sequence]

    def _create_kmers(self, sequence: List[str]) -> List[str]:
        tokens = []
        for i in range(0, len(sequence), self.k):
            kmer = ''.join(sequence[i:i + self.k])
            if len(kmer) < self.k:
                kmer += 'X' * (self.k - len(kmer))
            tokens.append(kmer)
        return tokens

    def _build_vocab(self, all_kmers: List[str]):
        unique_kmers = set(all_kmers)
        self.vocab = {kmer: idx for idx, kmer in enumerate(unique_kmers)}
        self.mask_id = len(self.vocab)

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]

    def _apply_random_masking(self, token_ids: List[int]) -> List[int]:
        masked_ids = []
        for token_id in token_ids:
            if random.random() < self.mask_prob:
                masked_ids.append(self.mask_id)
            else:
                masked_ids.append(token_id)
        return masked_ids

    def fit(self, snp_sequences: List[List[int]]):
        all_kmers = []

        for snp_seq in snp_sequences:
            letters = self._snp_to_letters(snp_seq)
            processed = self._preprocess_sequence(letters)
            kmers = self._create_kmers(processed)
            all_kmers.extend(kmers)

        self._build_vocab(all_kmers)

        print(f"Словарь создан. Размер: {len(self.vocab)} токенов")
        print(f"Mask ID: {self.mask_id}")

    def transform(self, snp_sequence: List[int]) -> Tuple[List[int], List[str]]:
        letters = self._snp_to_letters(snp_sequence)
        processed = self._preprocess_sequence(letters)
        tokens = self._create_kmers(processed)
        token_ids = self._tokens_to_ids(tokens)
        masked_ids = self._apply_random_masking(token_ids)

        return masked_ids, tokens

    def get_vocab_size(self) -> int:
        return len(self.vocab) + 1


def batch_process_snp_sequences(snp_sequences: List[List[int]], k: int = 6, mask_prob: float = 0.15):
    processor = SNP_to_signal_k_mer(k=k, mask_prob=mask_prob)
    processor.fit(snp_sequences)

    results = []
    for i, seq in enumerate(snp_sequences):
        masked_ids, tokens = processor.transform(seq)
        results.append({
            'sequence_id': i,
            'masked_ids': masked_ids,
            'tokens': tokens,
            'original_length': len(seq),
            'compressed_length': len(tokens),
            'compression_ratio': len(seq) / len(tokens),  # ИСПРАВЛЕНО: правильное название
            'masked_count': sum(1 for x in masked_ids if x == processor.mask_id),
            'masked_percentage': sum(1 for x in masked_ids if x == processor.mask_id) / len(masked_ids) * 100
        })

    return processor, results


def load_and_filter_data(file_path, prefix="Ca"):
    df = pd.read_csv(file_path)
    selected_columns = [col for col in df.columns if col.startswith(prefix)]
    new_df = df[selected_columns]
    return new_df


def analyze_batch_results(processor: SNP_to_signal_k_mer, results: List[Dict]):
    """Анализ результатов пакетной обработки"""
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


def print_detailed_examples(results: List[Dict], processor: SNP_to_signal_k_mer, num_examples: int = 3):
    """Подробный вывод примеров обработки"""
    print(f"\nДЕТАЛИ ОБРАБОТКИ (первые {num_examples} последовательностей):")
    print("=" * 60)

    for i in range(min(num_examples, len(results))):
        result = results[i]
        print(f"\n--- Последовательность #{result['sequence_id']} ---")
        print(f"Оригинальная длина: {result['original_length']} SNP")
        print(f"После k-mer: {result['compressed_length']} токенов")
        print(f"Коэффициент сжатия: {result['compression_ratio']:.2f}x")
        print(f"Замаскировано токенов: {result['masked_count']} ({result['masked_percentage']:.1f}%)")

        # Показываем первые 5 токенов
        print("\nПервые 5 токенов:")
        for j, (token, token_id) in enumerate(zip(result['tokens'][:5], result['masked_ids'][:5])):
            mask_status = "[MASK]" if token_id == processor.mask_id else ""
            original_id = processor.vocab[token] if token in processor.vocab else "N/A"
            print(f"  {j}: '{token}' -> ID: {original_id} | Masked: {token_id} {mask_status}")


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
            "sequence_id,original_length,compressed_length,compression_ratio,masked_count,masked_percentage,masked_ids,tokens\n")
        for result in results:
            # Преобразуем списки в строки для CSV
            masked_ids_str = ';'.join(map(str, result['masked_ids']))
            tokens_str = ';'.join(result['tokens'])
            f.write(
                f"{result['sequence_id']},{result['original_length']},{result['compressed_length']},{result['compression_ratio']:.2f},{result['masked_count']},{result['masked_percentage']:.1f},\"{masked_ids_str}\",\"{tokens_str}\"\n")

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
            for j, (token, token_id) in enumerate(zip(result['tokens'][:10], result['masked_ids'][:10])):
                mask_status = "[MASK]" if token_id == processor.mask_id else ""
                original_id = processor.vocab[token]
                f.write(f"  {j}: '{token}' -> Original ID: {original_id} | Masked ID: {token_id} {mask_status}\n")

    print(f"Файлы сохранены:")
    print(f"  - {csv_filename} (токенизированные данные)")
    print(f"  - {vocab_filename} (словарь)")
    print(f"  - {stats_filename} (статистика)")
    print(f"  - {binary_filename} (бинарные данные)")
    print(f"  - {examples_filename} (примеры токенов)")

    return results_dir

if __name__ == "__main__":
    print("ПАКЕТНАЯ ОБРАБОТКА SNP ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
    print("=" * 60)

    df = load_and_filter_data(file)
    num_samples = len(df)
    data_list = df.astype(int).values.tolist()

    # Пакетная обработка с параметрами из статьи
    print("\nЗапуск пакетной обработки...")
    processor, results = batch_process_snp_sequences(
        snp_sequences=data_list,
        k=6,  # 6-mer как в статье
        mask_prob=0.15  # 15% маскирование как в статье
    )

    # Анализ результатов
    analyze_batch_results(processor, results)

    results_dir = save_results_to_files(processor, results, "snp_batch_processing")



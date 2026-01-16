import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

# Определяем путь к файлу
file_path = 'snp_batch_processing_results_20251226_074153/processed_data.pkl'

# Проверяем существование файла
if not os.path.exists(file_path):
    print(f"Файл не найден: {file_path}")
    print("Проверьте путь к файлу")
    # Пробуем найти файл в текущей директории
    import glob

    pkl_files = glob.glob("**/*.pkl", recursive=True)
    if pkl_files:
        print(f"Найдены файлы .pkl: {pkl_files}")
        file_path = pkl_files[0]
        print(f"Используем файл: {file_path}")
    else:
        print("Не найдено ни одного .pkl файла")
        exit()


# Определяем класс для загрузки pickle
class SNP_to_signal_k_mer:
    def __init__(self, k=6, mask_prob=0.3):
        self.k = k
        self.mask_prob = mask_prob
        self.vocab = {}


# Загружаем данные
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Файл успешно загружен: {file_path}")
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    print("Пробуем загрузить с игнорированием ошибок...")

    import pickle
    import sys
    import types

    # Создаем временный модуль с нужным классом
    temp_module = types.ModuleType('temp_module')
    temp_module.SNP_to_signal_k_mer = SNP_to_signal_k_mer
    sys.modules['__main__'].SNP_to_signal_k_mer = SNP_to_signal_k_mer

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

# Изучаем структуру данных
print("\n" + "=" * 60)
print("АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
print("=" * 60)
print(f"Тип данных: {type(data)}")
print(f"Ключи в данных: {list(data.keys())}")

# ID маски (из вашего примера)
MASK_ID = 12109

# Извлекаем результаты
results = data['results']
print(f"\nТип results: {type(results)}")

# Изучаем структуру results
if isinstance(results, dict):
    print(f"Ключи в results: {list(results.keys())[:10]}...")  # Показываем первые 10 ключей
    print(f"Всего ключей в results: {len(results)}")

    # Пробуем найти примеры данных
    print("\nПримеры первых 5 записей в results:")
    for i, (key, value) in enumerate(list(results.items())[:5]):
        print(f"  Ключ {i}: {key[:50]}{'...' if len(str(key)) > 50 else ''}")
        print(f"  Значение тип: {type(value)}")
        if hasattr(value, '__dict__'):
            print(f"  Атрибуты: {list(vars(value).keys())[:5]}...")
        elif isinstance(value, dict):
            print(f"  Ключи значения: {list(value.keys())[:5]}...")
        print()

elif isinstance(results, list):
    print(f"Results - список длиной: {len(results)}")
    print("\nПример первой записи в results:")
    if len(results) > 0:
        print(f"  Тип: {type(results[0])}")
        if hasattr(results[0], '__dict__'):
            print(f"  Атрибуты: {list(vars(results[0]).keys())}")

        print(f"Result: {results[0]}")
        print(f"lenght of result: {len(results[0])}")
else:
    print("Неизвестный тип results")

# Параметры
parameters = data['parameters']
print(f"\nПараметры: {parameters}")


# Функция для извлечения токенов из результатов
def extract_tokens_from_results(results):
    """Извлекает токены и их ID из результатов"""
    tokens_data = []

    if isinstance(results, dict):
        # Если results - словарь
        for key, value in results.items():
            # Проверяем разные возможные структуры
            if isinstance(value, dict):
                # Если значение - словарь
                if 'token' in value and 'id' in value:
                    tokens_data.append({
                        'token': value['token'],
                        'id': value['id'],
                        'original_key': key
                    })
                elif 'sequence' in value and 'masked_id' in value:
                    tokens_data.append({
                        'token': value['sequence'],
                        'id': value['masked_id'],
                        'original_id': value.get('original_id'),
                        'original_key': key
                    })
                # Ищем другие возможные структуры
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and len(subvalue) == 6 and all(c in 'ATCGX' for c in subvalue):
                        # Нашли токен
                        tokens_data.append({
                            'token': subvalue,
                            'id': None,  # ID неизвестен
                            'original_key': f"{key}.{subkey}"
                        })

            elif hasattr(value, '__dict__'):
                # Если значение - объект
                obj_dict = vars(value)
                if 'token' in obj_dict and 'id' in obj_dict:
                    tokens_data.append({
                        'token': obj_dict['token'],
                        'id': obj_dict['id'],
                        'original_key': key
                    })
                # Ищем атрибуты с токенами
                for attr_name, attr_value in obj_dict.items():
                    if isinstance(attr_value, str) and len(attr_value) == 6 and all(c in 'ATCGX' for c in attr_value):
                        tokens_data.append({
                            'token': attr_value,
                            'id': None,
                            'original_key': f"{key}.{attr_name}"
                        })

    elif isinstance(results, list):
        # Если results - список
        for i, item in enumerate(results):
            if isinstance(item, dict):
                if 'token' in item and 'id' in item:
                    tokens_data.append({
                        'token': item['token'],
                        'id': item['id'],
                        'index': i
                    })
                elif 'sequence' in item and 'masked_id' in item:
                    tokens_data.append({
                        'token': item['sequence'],
                        'id': item['masked_id'],
                        'original_id': item.get('original_id'),
                        'index': i
                    })
            elif hasattr(item, '__dict__'):
                obj_dict = vars(item)
                if 'token' in obj_dict and 'id' in obj_dict:
                    tokens_data.append({
                        'token': obj_dict['token'],
                        'id': obj_dict['id'],
                        'index': i
                    })

    return tokens_data


# Извлекаем токены
print("\nИзвлечение токенов из results...")
tokens_data = extract_tokens_from_results(results)
print(f"Извлечено {len(tokens_data)} записей с токенами")

# Если токены не найдены, пробуем альтернативные методы
if len(tokens_data) == 0:
    print("\nТокены не найдены стандартным методом. Пробуем альтернативные подходы...")

    # Метод 1: Ищем токены в строковом представлении
    import re

    results_str = str(results)
    # Ищем последовательности из A, T, C, G, X длиной 6
    pattern = r'([ATCGX]{6})'
    found_tokens = re.findall(pattern, results_str)

    if found_tokens:
        unique_tokens = list(set(found_tokens))
        print(f"Найдено {len(unique_tokens)} уникальных токенов через regex")

        # Создаем фиктивные данные с токенами
        for token in unique_tokens[:1000]:  # Ограничим для производительности
            tokens_data.append({
                'token': token,
                'id': None,
                'source': 'regex'
            })

    # Метод 2: Ищем ID в строковом представлении
    id_pattern = r'id[:\s]*(\d+)'
    ids = re.findall(id_pattern, results_str.lower())
    if ids:
        print(f"Найдено {len(set(ids))} уникальных ID через regex")


    # Метод 3: Глубокий поиск во вложенных структурах
    def deep_search_for_tokens(obj, path=""):
        tokens = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                tokens.extend(deep_search_for_tokens(value, new_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                tokens.extend(deep_search_for_tokens(item, new_path))
        elif isinstance(obj, str) and len(obj) == 6 and all(c in 'ATCGX' for c in obj):
            tokens.append({
                'token': obj,
                'path': path,
                'id': None
            })

        return tokens


    deep_tokens = deep_search_for_tokens(results, "results")
    if deep_tokens:
        print(f"Найдено {len(deep_tokens)} токенов через глубокий поиск")
        tokens_data.extend(deep_tokens[:1000])  # Ограничим

# Создаем DataFrame
if tokens_data:
    df = pd.DataFrame(tokens_data)

    # Если есть ID, определяем маскированные токены
    if 'id' in df.columns and df['id'].notna().any():
        # Конвертируем ID в числовой формат
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df['is_masked'] = df['id'] == MASK_ID
        print(f"\nНайдено маскированных токенов (ID={MASK_ID}): {df['is_masked'].sum()}")
        print(f"Найдено немаскированных токенов: {(~df['is_masked']).sum()}")
    else:
        # Определяем по содержанию 'X' в токене
        df['is_masked'] = df['token'].str.contains('X')
        print(f"\nОпределение маскировки по содержанию 'X':")
        print(f"Маскированных токенов (содержат X): {df['is_masked'].sum()}")
        print(f"Немаскированных токенов: {(~df['is_masked']).sum()}")

    print(f"\nОбразец данных (первые 10 строк):")
    print(df.head(10))
else:
    print("\nНе удалось извлечь токены. Создаем пустой DataFrame.")
    df = pd.DataFrame(columns=['token', 'id', 'is_masked'])


# Функция для анализа статистики
def analyze_token_statistics(df, mask_id=MASK_ID):
    """Анализирует статистику токенов"""

    if len(df) == 0:
        return {
            'total_tokens': 0,
            'masked_tokens': 0,
            'unmasked_tokens': 0,
            'mask_percentage': 0,
            'position_counts': {},
            'pattern_counts': {},
            'masked_tokens_list': [],
            'unmasked_tokens_list': [],
            'id_distribution': {}
        }

    total_tokens = len(df)

    if 'is_masked' in df.columns:
        masked_tokens = df['is_masked'].sum()
        unmasked_tokens = total_tokens - masked_tokens
    else:
        masked_tokens = 0
        unmasked_tokens = total_tokens

    # Анализ маскированных токенов
    if masked_tokens > 0:
        masked_data = df[df['is_masked']]
        masked_tokens_list = masked_data['token'].tolist()
    else:
        masked_tokens_list = []

    if unmasked_tokens > 0:
        unmasked_tokens_list = df[~df['is_masked']]['token'].tolist() if 'is_masked' in df.columns else df[
            'token'].tolist()
    else:
        unmasked_tokens_list = []

    # Анализ позиций X в маскированных токенах
    mask_positions = []
    mask_patterns = []

    for token in masked_tokens_list:
        # Позиции X
        for i, char in enumerate(token):
            if char == 'X':
                mask_positions.append(i)

        # Паттерны масок
        pattern = ''.join(['X' if c == 'X' else 'O' for c in token])
        mask_patterns.append(pattern)

    position_counts = Counter(mask_positions)
    pattern_counts = Counter(mask_patterns)

    # Распределение ID
    id_distribution = {}
    if 'id' in df.columns and df['id'].notna().any():
        id_counts = df['id'].value_counts()
        id_distribution = id_counts.head(20).to_dict()  # Топ-20 ID

    return {
        'total_tokens': total_tokens,
        'masked_tokens': masked_tokens,
        'unmasked_tokens': unmasked_tokens,
        'mask_percentage': (masked_tokens / total_tokens * 100) if total_tokens > 0 else 0,
        'position_counts': position_counts,
        'pattern_counts': pattern_counts,
        'masked_tokens_list': masked_tokens_list,
        'unmasked_tokens_list': unmasked_tokens_list,
        'id_distribution': id_distribution,
        'unique_tokens': len(set(masked_tokens_list + unmasked_tokens_list))
    }


# Получаем статистику
stats = analyze_token_statistics(df, MASK_ID)


# Визуализация
def plot_comprehensive_statistics(stats, mask_id=MASK_ID):
    """Создает комплексную визуализацию статистики"""

    fig = plt.figure(figsize=(20, 15))

    # Общая сетка
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Круговая диаграмма
    ax1 = fig.add_subplot(gs[0, 0])
    if stats['total_tokens'] > 0:
        labels = ['Немаскированные', 'Маскированные']
        sizes = [stats['unmasked_tokens'], stats['masked_tokens']]
        colors = ['#66b3ff', '#ff9999']

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                           autopct=lambda p: f'{p:.1f}%\n({int(p * stats["total_tokens"] / 100)})',
                                           startangle=90)
        ax1.set_title(f'Распределение токенов\nВсего: {stats["total_tokens"]}',
                      fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        ax1.set_title('Распределение токенов')

    # 2. Столбчатая диаграмма
    ax2 = fig.add_subplot(gs[0, 1])
    if stats['total_tokens'] > 0:
        categories = ['Немаскированные', 'Маскированные']
        counts = [stats['unmasked_tokens'], stats['masked_tokens']]

        bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Количество токенов по типам', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Количество')
        ax2.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        ax2.set_title('Количество токенов')

    # 3. Распределение ID (если есть)
    ax3 = fig.add_subplot(gs[0, 2])
    if stats['id_distribution']:
        ids = list(stats['id_distribution'].keys())[:10]  # Топ-10 ID
        id_counts = [stats['id_distribution'][id] for id in ids]
        id_labels = [f'ID: {id}' for id in ids]

        # Подсвечиваем ID маски
        bar_colors = ['#ff9999' if id == mask_id else '#66b3ff' for id in ids]

        bars = ax3.bar(id_labels, id_counts, color=bar_colors, alpha=0.8, edgecolor='black')
        ax3.set_title(f'Топ-10 ID токенов\n(ID маски: {mask_id})', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Количество')
        ax3.set_xlabel('ID токена')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, id_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + max(id_counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Нет данных об ID', ha='center', va='center', fontsize=12)
        ax3.set_title('Распределение ID')

    # 4. Распределение позиций X
    ax4 = fig.add_subplot(gs[1, 0])
    if stats['position_counts']:
        positions = sorted(stats['position_counts'].keys())
        position_labels = [f'Поз. {pos + 1}' for pos in positions]
        counts = [stats['position_counts'][pos] for pos in positions]

        bars = ax4.bar(position_labels, counts, color='#ff9999', alpha=0.8, edgecolor='black')
        ax4.set_title('Распределение X по позициям', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Позиция в токене')
        ax4.set_ylabel('Количество X')
        ax4.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Нет маскированных токенов', ha='center', va='center', fontsize=12)
        ax4.set_title('Распределение X по позициям')

    # 5. Паттерны масок
    ax5 = fig.add_subplot(gs[1, 1])
    if stats['pattern_counts']:
        top_patterns = stats['pattern_counts'].most_common(8)
        patterns = [p[0] for p in top_patterns]
        counts = [p[1] for p in top_patterns]

        y_pos = np.arange(len(patterns))
        bars = ax5.barh(y_pos, counts, color='#ff9999', alpha=0.8, edgecolor='black')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(patterns)
        ax5.invert_yaxis()
        ax5.set_title('Топ-8 паттернов масок\n(X-маска, O-не маска)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Количество')
        ax5.grid(axis='x', alpha=0.3)

        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax5.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{count}', ha='left', va='center', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'Нет данных о паттернах', ha='center', va='center', fontsize=12)
        ax5.set_title('Паттерны масок')

    # 6. Распределение символов
    ax6 = fig.add_subplot(gs[1, 2])
    # Собираем все символы
    all_chars = []
    for token in stats['masked_tokens_list'] + stats['unmasked_tokens_list']:
        all_chars.extend(list(token))

    if all_chars:
        char_counts = Counter(all_chars)
        chars = sorted(char_counts.keys())
        counts = [char_counts[char] for char in chars]
        colors_chars = ['#1f77b4' if char in 'ATCG' else '#ff7f0e' for char in chars]

        bars = ax6.bar(chars, counts, color=colors_chars, alpha=0.8, edgecolor='black')
        ax6.set_title('Распределение символов', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Символ')
        ax6.set_ylabel('Количество')
        ax6.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'Нет данных о символах', ha='center', va='center', fontsize=12)
        ax6.set_title('Распределение символов')

    # 7. Примеры токенов (занимает 2 строки и все колонки)
    ax7 = fig.add_subplot(gs[2:, :])
    ax7.axis('off')

    # Получаем примеры
    masked_examples = stats['masked_tokens_list'][:20] if stats['masked_tokens'] > 0 else []
    unmasked_examples = stats['unmasked_tokens_list'][:20] if stats['unmasked_tokens'] > 0 else []

    text_content = "ПРИМЕРЫ ТОКЕНОВ:\n\n"

    if masked_examples:
        text_content += f"МАСКИРОВАННЫЕ токены (ID={mask_id}):\n"
        for i in range(0, len(masked_examples), 5):
            batch = masked_examples[i:i + 5]
            line = "  "
            for token in batch:
                # Подсвечиваем X
                colored_token = ''.join([f'[X]' if c == 'X' else c for c in token])
                line += f"{colored_token:10s} "
            text_content += line + "\n"

    if unmasked_examples:
        text_content += f"\nНЕМАСКИРОВАННЫЕ токены:\n"
        for i in range(0, len(unmasked_examples), 5):
            batch = unmasked_examples[i:i + 5]
            line = "  "
            for token in batch:
                line += f"{token:10s} "
            text_content += line + "\n"

    # Добавляем статистику
    text_content += f"\n\nСТАТИСТИКА:\n"
    text_content += f"  Всего токенов: {stats['total_tokens']}\n"
    text_content += f"  Уникальных токенов: {stats['unique_tokens']}\n"
    text_content += f"  Маскированных: {stats['masked_tokens']} ({stats['mask_percentage']:.1f}%)\n"
    text_content += f"  Немаскированных: {stats['unmasked_tokens']} ({100 - stats['mask_percentage']:.1f}%)\n"

    ax7.text(0.02, 0.98, text_content, transform=ax7.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    plt.suptitle(f'АНАЛИЗ ТОКЕНОВ ИЗ ФАЙЛА: {os.path.basename(file_path)}\nID маски: {mask_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()


# Строим графики
plot_comprehensive_statistics(stats, MASK_ID)

# Вывод подробной статистики
print("\n" + "=" * 80)
print("ПОДРОБНАЯ СТАТИСТИКА ТОКЕНОВ")
print("=" * 80)
print(f"Всего обработано записей: {stats['total_tokens']}")
print(f"Уникальных токенов: {stats['unique_tokens']}")
print(f"Процент уникальности: {(stats['unique_tokens'] / stats['total_tokens'] * 100):.1f}%" if stats[
                                                                                                    'total_tokens'] > 0 else "N/A")
print(f"Маскированных токенов: {stats['masked_tokens']} ({stats['mask_percentage']:.2f}%)")
print(f"Немаскированных токенов: {stats['unmasked_tokens']} ({100 - stats['mask_percentage']:.2f}%)")
print()

if stats['masked_tokens'] > 0:
    print("РАСПРЕДЕЛЕНИЕ 'X' ПО ПОЗИЦИЯМ В МАСКИРОВАННЫХ ТОКЕНАХ:")
    total_x = sum(stats['position_counts'].values())
    for pos in range(6):  # Для позиций 0-5
        count = stats['position_counts'].get(pos, 0)
        percentage = (count / total_x * 100) if total_x > 0 else 0
        print(f"  Позиция {pos + 1}: {count:4d} X ({percentage:5.1f}%)")

    print(f"\nВсего символов 'X': {total_x}")
    print(f"Среднее X на маскированный токен: {total_x / stats['masked_tokens']:.2f}" if stats[
                                                                                             'masked_tokens'] > 0 else "N/A")

    if stats['pattern_counts']:
        print("\nТОП-10 ПАТТЕРНОВ МАСКИРОВКИ:")
        for pattern, count in stats['pattern_counts'].most_common(10):
            percentage = (count / stats['masked_tokens']) * 100
            # Расшифровка паттерна
            decoded = pattern.replace('X', '[X]').replace('O', '_')
            print(f"  {decoded}: {count:4d} токенов ({percentage:5.1f}%)")

    if stats['masked_tokens_list']:
        print("\nТОП-10 САМЫХ ЧАСТЫХ МАСКИРОВАННЫХ ТОКЕНОВ:")
        masked_token_counts = Counter(stats['masked_tokens_list']).most_common(10)
        for token, count in masked_token_counts:
            # Находим ID если есть
            token_id = "N/A"
            if 'id' in df.columns and not df[df['token'] == token].empty:
                token_id = df[df['token'] == token]['id'].iloc[0]
            print(f"  {token}: {count:4d} раз (ID: {token_id})")

if stats['id_distribution']:
    print("\nТОП-10 САМЫХ ЧАСТЫХ ID:")
    for id_val, count in list(stats['id_distribution'].items())[:10]:
        is_mask = " [MASK]" if id_val == MASK_ID else ""
        print(f"  ID {id_val}: {count:4d} токенов{is_mask}")

print("\n" + "=" * 80)
print("СВОДКА ПО ФАЙЛУ:")
print("=" * 80)
print(f"Файл: {file_path}")
print(f"Размер файла: {os.path.getsize(file_path) / 1024:.1f} KB")
print(f"Дата модификации: {os.path.getmtime(file_path):.0f}")
print(f"Параметры обработки: {parameters}")

# Дополнительный анализ: сохраняем результаты в файл
output_file = "token_analysis_report.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ОТЧЕТ ПО АНАЛИЗУ ТОКЕНОВ\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Файл данных: {file_path}\n")
    f.write(f"ID маски: {MASK_ID}\n\n")

    f.write("СТАТИСТИКА:\n")
    f.write(f"  Всего токенов: {stats['total_tokens']}\n")
    f.write(f"  Уникальных токенов: {stats['unique_tokens']}\n")
    f.write(f"  Маскированных: {stats['masked_tokens']} ({stats['mask_percentage']:.2f}%)\n")
    f.write(f"  Немаскированных: {stats['unmasked_tokens']} ({100 - stats['mask_percentage']:.2f}%)\n\n")

    if stats['masked_tokens'] > 0:
        f.write("РАСПРЕДЕЛЕНИЕ 'X' ПО ПОЗИЦИЯМ:\n")
        for pos in range(6):
            count = stats['position_counts'].get(pos, 0)
            percentage = (count / sum(stats['position_counts'].values()) * 100) if sum(
                stats['position_counts'].values()) > 0 else 0
            f.write(f"  Позиция {pos + 1}: {count} X ({percentage:.1f}%)\n")

        f.write("\nТОП-10 МАСКИРОВАННЫХ ТОКЕНОВ:\n")
        for token, count in Counter(stats['masked_tokens_list']).most_common(10):
            f.write(f"  {token}: {count}\n")

    f.write("\nПРИМЕРЫ ТОКЕНОВ:\n")
    f.write("Маскированные:\n")
    for token in stats['masked_tokens_list'][:20]:
        f.write(f"  {token}\n")

    f.write("\nНемаскированные:\n")
    for token in stats['unmasked_tokens_list'][:20]:
        f.write(f"  {token}\n")

print(f"\nОтчет сохранен в файл: {output_file}")
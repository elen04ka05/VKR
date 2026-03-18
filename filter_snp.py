#ФИЛЬТР ФАЙЛА С SNP, ОСТАВЛЯЕМ ТОЛЬКО ИМЯ ОБРАЗЦА И "Ca"

import csv


def filter_snp_columns_with_samples(input_file, output_file):
    """Оставляет столбцы с SNP и первый столбец с именами образцов"""

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Читаем заголовок
        headers = next(reader)

        # Находим индексы столбцов
        snp_indices = []
        snp_headers = []

        # Всегда оставляем первый столбец (предполагаем, что это имена образцов)
        if headers:
            snp_indices.append(0)
            snp_headers.append(headers[0])

        # Добавляем столбцы, которые начинаются на "Ca"
        for i, header in enumerate(headers[1:], 1):  # Начинаем с индекса 1
            if header.startswith('Ca'):
                snp_indices.append(i)
                snp_headers.append(header)

        # Записываем отфильтрованный заголовок
        writer.writerow(snp_headers)

        # Обрабатываем строки
        row_count = 0
        for row in reader:
            # Оставляем только нужные столбцы
            filtered_row = [row[i] for i in snp_indices]
            writer.writerow(filtered_row)
            row_count += 1

            # Прогресс
            if row_count % 100 == 0:
                print(f"Обработано строк: {row_count}")

    print(f"\nФайл успешно отфильтрован!")
    print(f"Было столбцов: {len(headers)}")
    print(f"Осталось столбцов: {len(snp_headers)}")
    print(f"Обработано строк: {row_count}")
    print(f"Результат сохранен в {output_file}")


# Укажите ваши файлы
input_file = 'converted_snp_letters.csv'
output_file = 'snp_with_samples.csv'

filter_snp_columns_with_samples(input_file, output_file)
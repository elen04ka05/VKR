# ФИЛЬТРАЦИЯ ФАЙЛА ФЕНОТИПОВ, ОТБОР НУЖНОГО

import csv


def filter_snp_columns_with_samples(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        headers = next(reader)

        snp_indices = []

        if headers:
            snp_indices.append(0)

        for i, header in enumerate(headers[1:], 1):
            if header.startswith('Аскохитоз (поражение)'):
                snp_indices.append(i)

        new_headers = ['SNP_ID', 'Ascochytosis']
        writer.writerow(new_headers)

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
    print(f"Осталось столбцов: {len(new_headers)}")
    print(f"Обработано строк: {row_count}")
    print(f"Результат сохранен в {output_file}")


input_file = 'pheno_2016_VIRVFVIR_421_408_synchro.csv'
output_file = 'pheno_ascoh.csv'

filter_snp_columns_with_samples(input_file, output_file)

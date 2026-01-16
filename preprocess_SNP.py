import csv


def create_final_snp_matrix(input_file, output_file):
    """Создает финальную матрицу SNP: без заголовков, без ID, только X"""

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)

        # Пропускаем заголовок
        headers = next(reader)

        # Обрабатываем строки
        row_count = 0
        for row in reader:
            # Пропускаем первый столбец (ID образцов)
            # Заменяем все SNP-буквы на X
            snp_values = []
            for value in row[1:]:  # Пропускаем первый столбец
                if value.strip() == "A":
                    snp_values.append("A")
                elif value.strip() == "T":
                    snp_values.append("T")
                elif value.strip() == "C":
                    snp_values.append("C")
                elif value.strip() == "G":
                    snp_values.append("G")
                elif value.strip() == "W" or value.strip() == "M" or value.strip() == "R" or value.strip() == "Y" or value.strip() == "S" or value.strip() == "K":
                    snp_values.append("X")

            # Записываем строку как CSV
            outfile.write(','.join(snp_values) + '\n')
            row_count += 1

            if row_count % 100 == 0:
                print(f"Обработано строк: {row_count}")

    print(f"\nФинальная матрица SNP создана!")
    print(f"Размер матрицы: {row_count} строк × {len(headers) - 1} столбцов (SNP)")
    print(f"Файл сохранен: {output_file}")


# Укажите ваши файлы
input_file = 'snp_with_samples.csv'  # Ваш файл только с SNP
output_file = 'final_snp_matrix.csv'

create_final_snp_matrix(input_file, output_file)
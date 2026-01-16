import csv

# Карта замен
SNP_MAP = {
    'A_T': {0: 'A', 1: 'W', 2: 'T'},
    'A_C': {0: 'A', 1: 'M', 2: 'C'},
    'A_G': {0: 'A', 1: 'R', 2: 'G'},
    'C_T': {0: 'C', 1: 'Y', 2: 'T'},
    'C_G': {0: 'C', 1: 'S', 2: 'G'},
    'G_T': {0: 'G', 1: 'K', 2: 'T'},
    'T_A': {0: 'T', 1: 'W', 2: 'A'},
    'C_A': {0: 'C', 1: 'M', 2: 'A'},
    'G_A': {0: 'G', 1: 'R', 2: 'A'},
    'T_C': {0: 'T', 1: 'Y', 2: 'C'},
    'G_C': {0: 'G', 1: 'S', 2: 'C'},
    'T_G': {0: 'T', 1: 'K', 2: 'G'},
}


def get_snp_type(snp_id):
    """Определяет тип SNP из его названия"""
    parts = snp_id.split('_')
    if len(parts) >= 4:
        allele1 = parts[-2]
        allele2 = parts[-1]
        return f"{allele1}_{allele2}"
    return None


def convert_value(value, snp_type):
    """Конвертирует числовое значение в буквенное обозначение"""
    if value == '' or value is None:
        return ''

    str_value = str(value).strip()
    if not str_value.isdigit():
        return str_value

    int_value = int(str_value)

    if snp_type and snp_type in SNP_MAP:
        if int_value in SNP_MAP[snp_type]:
            return SNP_MAP[snp_type][int_value]

    # Стандартная логика
    if int_value == 0:
        if snp_type:
            return snp_type.split('_')[0]
        return '0'
    elif int_value == 2:
        if snp_type:
            return snp_type.split('_')[1]
        return '2'
    elif int_value == 1:
        # Находим подходящее обозначение для гетерозигот
        for snp_key in ['A_T', 'A_C', 'A_G', 'C_T', 'C_G', 'G_T']:
            if 1 in SNP_MAP[snp_key]:
                return SNP_MAP[snp_key][1]
        return '1'

    return str_value


# Обработка файла
input_file = 'total_df_for_aio_chickpea_28042016_synchro.csv'
output_file = 'converted_snp_letters.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Читаем заголовок
    headers = next(reader)
    writer.writerow(headers)

    # Определяем типы SNP для каждого столбца
    snp_types = []
    for header in headers:
        if header.startswith('Ca'):
            snp_type = get_snp_type(header)
            snp_types.append(snp_type)
        else:
            snp_types.append(None)

    # Обрабатываем строки
    row_count = 0
    for row in reader:
        new_row = []
        for i, value in enumerate(row):
            if i < len(snp_types):
                new_value = convert_value(value, snp_types[i])
                new_row.append(new_value)
            else:
                new_row.append(value)
        writer.writerow(new_row)
        row_count += 1

        # Прогресс
        if row_count % 100 == 0:
            print(f"Обработано строк: {row_count}")

print(f"\nГотово! Обработано {row_count} строк.")
print(f"Результат сохранен в файл: {output_file}")
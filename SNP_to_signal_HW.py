import pandas as pd
import numpy as np

file = "total_df_for_aio_chickpea_28042016_synchro.csv"

def load_and_filter_data(file_path, prefix="Ca"):
    df = pd.read_csv(file_path)
    selected_columns = [col for col in df.columns if col.startswith(prefix)]
    return selected_columns

def frequency_calculation(file_path, SNP):
    df = pd.read_csv(file_path)
    selected_data = df[SNP].copy()

    hw_data = {}

    for snp in SNP:
        try:
            snp_values = selected_data[snp]

            D = np.sum(snp_values == 0)  # AA
            H = np.sum(snp_values == 1)  # Aa
            N = len(snp_values)  # всего образцов

            p = (2*D + H)/(2*N)
            q = 1- p

            p_2 = p * p
            _2_p_q = 2 * p * q
            q_2 = q * q

            hw_values = snp_values.replace({0: p_2, 1: _2_p_q, 2: q_2})
            hw_data[snp] = hw_values

            #print(f"Обработан снип {snp}: p={p:.3f}, p²={p_2:.3f}, 2pq={_2_p_q:.3f}, q²={q_2:.3f}")
        except KeyError:
            print(f"Предупреждение: снип {snp} не найден в данных")
        except Exception as e:
            print(f"Ошибка при обработке снипа {snp}: {e}")
    return pd.DataFrame(hw_data)

df_hw_selected = frequency_calculation(file, load_and_filter_data(file))

#print(df_hw_selected)

df_hw_selected.to_csv("hw_encoded_snps.csv")
print("Данные сохранены в hw_encoded_snps.csv")
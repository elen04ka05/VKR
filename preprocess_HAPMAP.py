import pandas as pd
import csv

hapmap = pd.read_csv("my_snps_letters.hmp.txt", sep="\t")

genotype_data = hapmap.iloc[:, 11:]

len = len(genotype_data.columns)
#print(len)
#print(genotype_data)
#print(hapmap)

new_genotype = []
for i in range(2, genotype_data.shape[1]):  # колонки 11 и дальше
    sample_name = genotype_data.columns[i]
    genotypes = genotype_data.iloc[:, i]

    print(f"Образец: {sample_name}")
    print(f"  Всего генотипов: {genotypes.shape[0]}")
    #print(genotypes)

    sequence = []
    for genotype in genotypes:
        if genotype == "A": sequence.append("A")
        elif genotype == "T": sequence.append("T")
        elif genotype == "C": sequence.append("C")
        elif genotype == "G": sequence.append("G")
        elif genotype == "W" or genotype == "M" or genotype == "R" or genotype == "Y" or genotype == "S" or genotype == "K": sequence.append("X")


    new_genotype.append(sequence)
    #if i == 2:
        #print(sequence)

#print(new_genotype[0])


df = pd.DataFrame(new_genotype)
df.to_csv('output_SNP_HAPMAP.csv', index=False, header=False)

print("Данные успешно записаны в samples.csv")
print(f"Размерность: {len(new_genotype)} образцов, {len(new_genotype[0])} признаков")

'''with open('output_SNP_HAPMAP.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(new_genotype)'''

'''file = pd.read_csv("output_SNP_HAPMAP.csv", sep="\t")
print(file)'''
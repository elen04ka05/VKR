'''import torch

# Загружаем эмбеддинги
embeddings = torch.load("genomic_embeddings.pt")

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")
print(f"Number of sequences: {embeddings.shape[0]}")
print(f"Sequence length: {embeddings.shape[1]}")
print(f"Embedding dimension: {embeddings.shape[2]}")
print()

# Статистика по эмбеддингам
print(f"Min value: {embeddings.min():.4f}")
print(f"Max value: {embeddings.max():.4f}")
print(f"Mean value: {embeddings.mean():.4f}")
print(f"Std value: {embeddings.std():.4f}")
print()

# Пример эмбеддингов для первой последовательности
print("First sequence embeddings (first 5 positions, first 5 dimensions):")
print(embeddings[0, :5, :5])'''

'''import torch

# Загружаем файл
data = torch.load("genomic_mlm_data.pt")

# Смотрим структуру
print("Keys in data:", data.keys())
print()

# Информация о vocab
print(f"Vocab size: {len(data['vocab'])}")
print("First 10 vocab items:")
for i, (token, idx) in enumerate(list(data['vocab'].items())[:10]):
    print(f"  {token}: {idx}")
print()

# Информация о батчах
print(f"Number of batches: {len(data['batches'])}")
print()

# Смотрим первый батч
first_batch = data['batches'][0]
print("First batch keys:", first_batch.keys())
print(f"input_ids shape: {first_batch['input_ids'].shape}")
print(f"labels shape: {first_batch['labels'].shape}")
print(f"attention_mask shape: {first_batch['attention_mask'].shape}")
print()

# Пример данных из первого батча
print("First sequence input_ids (first 20 tokens):")
print(first_batch['input_ids'][0][:20])
print()
print("First sequence labels (first 20 tokens):")
print(first_batch['labels'][0][:20])
print()
print("First sequence attention_mask (first 20 tokens):")
print(first_batch['attention_mask'][0][:20])'''

import torch
import sys
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Проверка CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
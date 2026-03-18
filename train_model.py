'''import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        """
        x: [B, T, D]
        attention_mask: [B, T] (1 = valid, 0 = pad)
        """

        # ---- Self-Attention ----
        x_norm = self.norm1(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
        )

        x = x + attn_out

        # ---- Feed Forward ----
        x = x + self.ffn(self.norm2(x))

        return x

class GenomicTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 4,
        proj_dim: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Step 7: Linear projector
        self.projector = nn.Linear(d_model, proj_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, T]
        attention_mask: [B, T] (optional)
        """

        # ---- Embedding + Position ----
        x = self.embedding(input_ids)      # [B, T, D]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # ---- Encoder layers ----
        for layer in self.layers:
            x = layer(x, attention_mask)

        # ---- Projection ----
        x = self.projector(x)              # [B, T, proj_dim]

        return x

class MLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: [B, T, hidden_dim]
        """
        return self.linear(x)

class PhenotypeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """
        x = x.flatten(1)  # [B, T*D]
        return self.mlp(x)

data = torch.load("genomic_mlm_data.pt")
vocab_size = len(data["vocab"])
pad_id = data["pad_id"]
max_seq_len = max(
    batch["input_ids"].shape[1]
    for batch in data["batches"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = GenomicTransformerEncoder(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    pad_id=pad_id,
)

mlm_head = MLMHead(
    hidden_dim=64,
    vocab_size=vocab_size,
)
encoder = encoder.to(device)
mlm_head = mlm_head.to(device)

encoder.eval()
all_embeddings = []

print("Generating embeddings for all batches...")
with torch.no_grad():
    for i, batch in enumerate(data["batches"]):
        # Перемещаем данные на устройство
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Получаем эмбеддинги
        out = encoder(input_ids, attention_mask)
        all_embeddings.append(out.cpu())

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data['batches'])} batches")

embeddings_tensor = torch.cat(all_embeddings, dim=0)
torch.save(embeddings_tensor, "genomic_embeddings.pt")
print(f"Saved embeddings with shape: {embeddings_tensor.shape}")

# Сохраняем модель
torch.save(
    {
        "encoder": encoder.state_dict(),
        "mlm_head": mlm_head.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "d_model": 128,
            "n_heads": 8,
            "d_ff": 512,
            "num_layers": 4,
            "proj_dim": 64,
        }
    },
    "genomic_transformer_model.pt"
)
print("Model saved successfully!")'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        """
        x: [B, T, D]
        attention_mask: [B, T] (1 = valid, 0 = pad)
        """

        # ---- Self-Attention ----
        x_norm = self.norm1(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
        )

        x = x + attn_out

        # ---- Feed Forward ----
        x = x + self.ffn(self.norm2(x))

        return x


class GenomicTransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int = 128,
            n_heads: int = 8,
            d_ff: int = 512,
            num_layers: int = 4,
            proj_dim: int = 64,
            dropout: float = 0.1,
            pad_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.projector = nn.Linear(d_model, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, T]
        attention_mask: [B, T] (optional)
        """

        # ---- Embedding + Position ----
        x = self.embedding(input_ids)  # [B, T, D]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # ---- Encoder layers ----
        for layer in self.layers:
            x = layer(x, attention_mask)

        # ---- Projection ----
        x = self.projector(x)  # [B, T, proj_dim]

        return x


class MLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: [B, T, hidden_dim]
        """
        return self.linear(x)

class MaskedSequencesDataset(Dataset):
    """
    Загружает данные из CSV файла, где каждая строка - это последовательность ID с масками
    """

    def __init__(self, csv_file_path, max_seq_len=None, pad_id=0):
        """
        Args:
            csv_file_path: путь к CSV файлу с последовательностями
            max_seq_len: максимальная длина последовательности (обрезаем/дополняем)
            pad_id: ID для заполнения (padding)
        """
        print(f"Загружаю данные из: {csv_file_path}")

        # Читаем CSV файл
        # В файле каждая строка: "id1,id2,id3,..." без заголовков
        self.data = []

        with open(csv_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Убираем пробелы и переносы строк, разбиваем по запятым
                ids_str = line.strip().split(',')
                # Преобразуем в числа
                ids = [int(x) for x in ids_str if x]  # if x на случай пустых строк
                self.data.append(ids)

        print(f"Загружено {len(self.data)} последовательностей")
        print(f"Пример первой последовательности (первые 20): {self.data[0][:20]}")

        # Определяем реальную максимальную длину
        real_max_len = max(len(seq) for seq in self.data)
        print(f"Реальная максимальная длина: {real_max_len}")

        # Устанавливаем длину для паддинга
        if max_seq_len is None:
            self.max_seq_len = real_max_len
        else:
            self.max_seq_len = min(max_seq_len, real_max_len)

        print(f"Используемая длина (после обрезки/дополнения): {self.max_seq_len}")

        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Берем последовательность
        seq = self.data[idx]

        # Обрезаем, если слишком длинная
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        # Создаем input_ids (сами ID)
        input_ids = seq.copy()

        # Создаем attention_mask (1 для реальных токенов, 0 для паддинга)
        attention_mask = [1] * len(input_ids)

        # Дополняем до нужной длины (padding)
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        # Преобразуем в тензоры
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


# ============= ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ =============

def train_model(
        csv_file_path,
        vocab_size,
        pad_id=0,
        mask_id=None,
        batch_size=32,
        epochs=10,
        learning_rate=1e-4,
        d_model=128,
        n_heads=8,
        d_ff=512,
        num_layers=4,
        proj_dim=64,
        max_seq_len=None,  # если None, берется максимальная длина из данных
        save_path="genomic_model.pt"
):
    """
    Обучает модель на данных из CSV файла
    """

    # ---- Устройство ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Загружаем данные ----
    dataset = MaskedSequencesDataset(
        csv_file_path=csv_file_path,
        max_seq_len=max_seq_len,
        pad_id=pad_id
    )

    # Создаем DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # для Windows лучше 0
    )

    # Получаем реальную максимальную длину
    actual_max_seq_len = dataset.max_seq_len

    # ---- Создаем модель ----
    print("\nСоздаю модель...")
    encoder = GenomicTransformerEncoder(
        vocab_size=vocab_size,
        max_seq_len=actual_max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        proj_dim=proj_dim,
        dropout=0.1,
        pad_id=pad_id
    )

    # Если есть mask_id, создаем head для MLM
    if mask_id is not None:
        mlm_head = MLMHead(hidden_dim=proj_dim, vocab_size=vocab_size)
        mlm_head = mlm_head.to(device)

    encoder = encoder.to(device)

    # ---- Оптимизатор ----
    if mask_id is not None:
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(mlm_head.parameters()),
            lr=learning_rate
        )
    else:
        optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # ---- Цикл обучения ----
    print(f"\nНачинаю обучение на {len(dataset)} образцах...")
    print(f"Batch size: {batch_size}, Всего батчей: {len(dataloader)}")

    for epoch in range(epochs):
        total_loss = 0
        encoder.train()
        if mask_id is not None:
            mlm_head.train()

        # Прогресс-бар для батчей
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Прямой проход
            embeddings = encoder(input_ids, attention_mask)

            if mask_id is not None:
                # Если обучаем с маскированием (MLM)
                logits = mlm_head(embeddings)

                # Создаем targets (для MLM нужно предсказывать исходные ID)
                # В этом случае input_ids уже содержат маски, но нам нужны оригиналы
                # Предполагаем, что в данных уже есть маски, и мы учимся их восстанавливать
                targets = input_ids.clone()

                # Считаем loss только для замаскированных позиций
                mask_positions = (input_ids == mask_id)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                    reduction='none'
                )
                loss = (loss * mask_positions.view(-1).float()).sum() / (mask_positions.sum().float() + 1e-8)
            else:
                # Простое автоэнкодерное обучение (восстанавливаем вход)
                # В этом случае можно использовать MSE или другой loss
                # Но для простоты пока пропустим
                loss = torch.tensor(0.0).to(device)
                print("Внимание: без mask_id loss не определен!")

            # Обратное распространение
            if mask_id is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Обновляем прогресс-бар
            pbar.set_postfix({'loss': f'{loss.item():.4f}' if mask_id is not None else 'N/A'})

        # Выводим статистику за эпоху
        if mask_id is not None:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} finished. Average loss: {avg_loss:.4f}")

    # ---- Сохраняем модель ----
    print(f"\nСохраняю модель в {save_path}")

    # Сначала получаем эмбеддинги для всего датасета (если нужно)
    encoder.eval()
    all_embeddings = []

    print("Генерирую эмбеддинги для всех образцов...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            out = encoder(input_ids, attention_mask)
            all_embeddings.append(out.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    embeddings_path = save_path.replace('.pt', '_embeddings.pt')
    torch.save(embeddings_tensor, embeddings_path)
    print(f"Сохранены эмбеддинги: {embeddings_tensor.shape} -> {embeddings_path}")

    # Сохраняем модель
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "mlm_head": mlm_head.state_dict() if mask_id is not None else None,
            "config": {
                "vocab_size": vocab_size,
                "max_seq_len": actual_max_seq_len,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "num_layers": num_layers,
                "proj_dim": proj_dim,
                "pad_id": pad_id,
                "mask_id": mask_id
            }
        },
        save_path
    )
    print(f"Модель сохранена: {save_path}")

    return encoder, embeddings_tensor


# ============= ЗАПУСК ОБУЧЕНИЯ =============

if __name__ == "__main__":
    # Параметры
    CSV_FILE = "snp_batch_processing_results_20260312_115114/masked_sequences_ids.csv"  # твой файл с ID после маскирования
    VOCAB_SIZE = 12110  # УКАЖИ РАЗМЕР СВОЕГО СЛОВАРЯ (len(vocab) + 1 для маски)
    MASK_ID = 12109  # УКАЖИ ID МАСКИ (обычно vocab_size - 1)
    PAD_ID = 0  # обычно 0

    # Запускаем обучение
    encoder, embeddings = train_model(
        csv_file_path=CSV_FILE,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        mask_id=MASK_ID,  # если передать None, будет обучение без MLM
        batch_size=32,
        epochs=10,
        learning_rate=1e-4,
        d_model=128,
        n_heads=8,
        d_ff=512,
        num_layers=4,
        proj_dim=64,
        save_path="my_genomic_model.pt"
    )


import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

torch.set_num_threads(2)


class PositionalEncoding(nn.Module): #позиционное кодирование, пока без него
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
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

    def forward(self, x):

        x_norm = self.norm1(x)

        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
        )

        x = x + attn_out

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
            use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        if use_positional_encoding:
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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.projector(x)
        return x


class AscochytaClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class GenomicDataset(Dataset):
    def __init__(self, sequences_file, phenotypes_file, max_seq_len=None, pad_id=0):
        print(f"Загружаю последовательности из: {sequences_file}")
        print(f"Загружаю фенотипы из: {phenotypes_file}")

        self.sequences = []
        with open(sequences_file, 'r', encoding='utf-8') as f:
            for line in f:
                ids_str = line.strip().split(',')
                ids = [int(x) for x in ids_str if x]
                self.sequences.append(ids)

        phenotypes_df = pd.read_csv(phenotypes_file, encoding='utf-8')

        print(f"Колонки в файле фенотипов: {phenotypes_df.columns.tolist()}")
        self.labels = []

        label_dict = dict(zip(phenotypes_df.iloc[:, 0].astype(str), phenotypes_df.iloc[:, 1]))

        for i in range(len(self.sequences)):
            snp_id = str(i)
            if snp_id in label_dict:
                self.labels.append(label_dict[snp_id])
            else:
                if i < len(phenotypes_df):
                    self.labels.append(phenotypes_df.iloc[i, 1])
                else:
                    raise ValueError(f"Нет метки для последовательности {i}")

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        print(f"Загружено {len(self.sequences)} последовательностей")
        print(f"Загружено {len(self.labels)} меток")
        print(f"Распределение меток: уникальные значения = {torch.unique(self.labels).tolist()}")

        # преобразуем метки 1,3,5,7 в 0,1,2,3
        label_mapping = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}
        self.labels = torch.tensor([label_mapping[label.item()] for label in self.labels], dtype=torch.long)

        print(f"Метки после преобразования: уникальные значения = {torch.unique(self.labels).tolist()}")

        real_max_len = max(len(seq) for seq in self.sequences)
        print(f"Реальная максимальная длина последовательности: {real_max_len}")

        if max_seq_len is None:
            self.max_seq_len = real_max_len
        else:
            self.max_seq_len = min(max_seq_len, real_max_len)

        print(f"Используемая длина: {self.max_seq_len}")

        self.pad_id = pad_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        input_ids = seq.copy()

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)

        label = self.labels[idx]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': label
        }


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График потерь
    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # График точности
    axes[1].plot(train_accs, label='Train Accuracy', marker='o')
    axes[1].plot(val_accs, label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Графики обучения сохранены в {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                          save_path='confusion_matrix.png'):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    print(f"y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        print("ОШИБКА: Нет данных для построения матрицы ошибок!")
        return

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Матрица ошибок сохранена в {save_path}")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"\nДетальные метрики:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

    print(f"\nМетрики по классам:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        prec = precision_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        rec = recall_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        f1 = f1_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        print(f"{name:<10} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")


def train_classifier(
        sequences_file,
        phenotypes_file,
        vocab_size,
        pad_id=0,
        batch_size=16,
        epochs=20,
        learning_rate=1e-4,
        d_model=128,
        n_heads=8,
        d_ff=512,
        num_layers=4,
        proj_dim=64,
        num_classes=4,
        use_positional_encoding=True,
        save_path="ascochyta_classifier.pt",
        plots_dir=None,
        device=None
):
    if plots_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pe_status = "with_pe" if use_positional_encoding else "without_pe"
        plots_dir = f"training_plots_{timestamp}_{pe_status}"

    os.makedirs(plots_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Use positional encoding: {use_positional_encoding}")

    dataset = GenomicDataset(
        sequences_file=sequences_file,
        phenotypes_file=phenotypes_file,
        max_seq_len=None,
        pad_id=pad_id
    )

    # Анализ распределения классов
    all_labels = [dataset[i]['label'] for i in range(len(dataset))]
    print(f"\n=== РАСПРЕДЕЛЕНИЕ КЛАССОВ ===")
    for i in range(num_classes):
        count = sum(1 for l in all_labels if l == i)
        print(f"Class {i}: {count:4d} samples ({count / len(all_labels) * 100:5.1f}%)")

    # Разделяем на train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Создаем модель
    encoder = GenomicTransformerEncoder(
        vocab_size=vocab_size,
        max_seq_len=dataset.max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        proj_dim=proj_dim,
        dropout=0.1,
        pad_id=pad_id,
        use_positional_encoding=use_positional_encoding
    )

    classifier = AscochytaClassifier(
        input_dim=proj_dim,
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.1
    )

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    # Оптимизатор
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nНачинаю обучение...")

    for epoch in range(epochs):
        # Training
        encoder.train()
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device).long()

            if len(labels.shape) > 1:
                labels = labels.squeeze()

            # Forward
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)  # ← для 4 классов
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(
                {'loss': f'{loss.item():.4f}', 'acc': f'{train_correct / train_total:.3f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)


        encoder.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_preds = []


        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device).long().squeeze()

                embeddings = encoder(input_ids)
                logits = classifier(embeddings)

                loss = criterion(logits, labels)

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(predictions.cpu().numpy())



        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'max_seq_len': dataset.max_seq_len,
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'd_ff': d_ff,
                    'num_layers': num_layers,
                    'proj_dim': proj_dim,
                    'pad_id': pad_id,
                    'num_classes': num_classes
                }
            }, save_path)
            print(f"✓ Saved best model with val_acc: {val_acc:.4f}")

    print(f"\nОбучение завершено! Лучшая модель сохранена в {save_path}")
    print(f"Лучшая валидационная точность: {best_val_acc:.4f}")

    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(plots_dir, 'training_history.png')
    )

    print("\nОценка лучшей модели на валидационной выборке...")

    checkpoint = torch.load(save_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    encoder.eval()
    classifier.eval()

    val_all_labels = []
    val_all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device).long().squeeze()

            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)

            val_all_labels.extend(labels.cpu().numpy())
            val_all_preds.extend(predictions.cpu().numpy())

            if batch_idx == 0:
                print(f"Debug - batch {batch_idx}:")
                print(f"  labels shape: {labels.shape}")
                print(f"  predictions shape: {predictions.shape}")
                print(f"  labels sample: {labels[:5].cpu().numpy()}")
                print(f"  predictions sample: {predictions[:5].cpu().numpy()}")

    print(f"\nСобрано {len(val_all_labels)} меток и {len(val_all_preds)} предсказаний")

    if len(val_all_labels) == 0 or len(val_all_preds) == 0:
        print("ОШИБКА: Не удалось собрать предсказания!")
    else:
        val_all_labels = np.array(val_all_labels).flatten()
        val_all_preds = np.array(val_all_preds).flatten()

        print(f"Тип val_all_preds после конвертации: {type(val_all_preds)}")
        print(f"Shape val_all_preds: {val_all_preds.shape}")
        print(f"Уникальные метки: {np.unique(val_all_labels)}")
        print(f"Уникальные предсказания: {np.unique(val_all_preds)}")

        plot_confusion_matrix(
            val_all_labels, val_all_preds,
            class_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
            save_path=os.path.join(plots_dir, 'confusion_matrix.png')
        )

    print("\nClassification Report:")
    print(classification_report(val_all_labels,
                                val_all_preds,
                                target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4']))

    return encoder, classifier, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss
    }


# ============= ЗАПУСК ОБУЧЕНИЯ =============

if __name__ == "__main__":
    SEQUENCES_FILE = "snp_batch_processing_results_20260312_115114/masked_sequences_ids.csv"
    PHENOTYPES_FILE = "pheno_ascoh.csv"
    VOCAB_SIZE = 12110
    PAD_ID = 0

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    encoder, classifier, history = train_classifier(
        sequences_file=SEQUENCES_FILE,
        phenotypes_file=PHENOTYPES_FILE,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        batch_size=8,
        epochs=50,
        learning_rate=1e-4,
        d_model=64,
        n_heads=4,
        d_ff=256,
        num_layers=2,
        proj_dim=32,
        num_classes=4,
        use_positional_encoding=False,
        save_path="ascochyta_transformer.pt",
        plots_dir=None
    )

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Лучшая точность на валидации: {history['best_val_acc']:.4f}")
    print(f"Графики сохранены в папке: ascochyta_training_plots/")

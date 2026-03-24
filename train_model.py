import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


torch.set_num_threads(2)

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


class AscochytaClassifier(nn.Module):
    """
    Классификационная голова для предсказания аскохитоза
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()

        # Используем среднее по всем токенам (mean pooling) для получения вектора всей последовательности
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, attention_mask=None):
        """
        x: [B, T, D] - эмбеддинги от энкодера
        attention_mask: [B, T] - маска внимания
        """
        if attention_mask is not None:
            # Учитываем только реальные токены (не паддинг)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            sum_embeddings = (x * mask_expanded).sum(dim=1)  # [B, D]
            sum_mask = mask_expanded.sum(dim=1)  # [B, 1]
            pooled = sum_embeddings / (sum_mask + 1e-8)  # среднее только по реальным токенам
        else:
            # Простое среднее по всем токенам
            pooled = x.mean(dim=1)  # [B, D]

        return self.classifier(pooled)  # [B, num_classes]


class GenomicDataset(Dataset):
    """
    Датасет для обучения с метками аскохитоза
    """

    def __init__(self, sequences_file, phenotypes_file, snp_id_col='SNP_ID', phenotype_col='Ascochytosis',
                 max_seq_len=None, pad_id=0):
        """
        Args:
            sequences_file: CSV файл с последовательностями ID (masked_sequences_ids.csv)
            phenotypes_file: CSV файл с фенотипами (pheno_ascoh_renamed.csv)
            snp_id_col: название колонки с ID SNP в файле фенотипов
            phenotype_col: название колонки с аскохитозом
            max_seq_len: максимальная длина последовательности
            pad_id: ID для паддинга
        """
        print(f"Загружаю последовательности из: {sequences_file}")
        print(f"Загружаю фенотипы из: {phenotypes_file}")

        # Загружаем последовательности
        self.sequences = []
        with open(sequences_file, 'r', encoding='utf-8') as f:
            for line in f:
                ids_str = line.strip().split(',')
                ids = [int(x) for x in ids_str if x]
                self.sequences.append(ids)

        # Загружаем фенотипы
        phenotypes_df = pd.read_csv(phenotypes_file, encoding='utf-8')

        # Проверяем, что у нас есть все нужные колонки
        print(f"Колонки в файле фенотипов: {phenotypes_df.columns.tolist()}")

        # Предполагаем, что в файле фенотипов:
        # - первый столбец - SNP_ID (соответствует индексам последовательностей)
        # - второй столбец - Ascochytosis (значения аскохитоза)
        self.labels = []

        # Создаем словарь для быстрого поиска меток
        label_dict = dict(zip(phenotypes_df.iloc[:, 0].astype(str), phenotypes_df.iloc[:, 1]))

        # Для каждой последовательности находим соответствующую метку
        # Предполагаем, что последовательности соответствуют строкам в том же порядке, что и в файле фенотипов
        # Но для надежности используем индексы строк
        for i in range(len(self.sequences)):
            # Используем индекс строки как ключ
            snp_id = str(i)  # или phenotypes_df.iloc[i, 0] если есть конкретные ID
            if snp_id in label_dict:
                self.labels.append(label_dict[snp_id])
            else:
                # Если не нашли, используем метку из той же строки по порядку
                if i < len(phenotypes_df):
                    self.labels.append(phenotypes_df.iloc[i, 1])
                else:
                    raise ValueError(f"Нет метки для последовательности {i}")

        # Преобразуем метки в числа
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        print(f"Загружено {len(self.sequences)} последовательностей")
        print(f"Загружено {len(self.labels)} меток")
        print(f"Распределение меток: уникальные значения = {torch.unique(self.labels).tolist()}")

        # ⭐ НОВЫЙ КОД: преобразуем метки 1,3,5,7 в 0,1,2,3
        label_mapping = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}
        self.labels = torch.tensor([label_mapping[label.item()] for label in self.labels], dtype=torch.long)

        print(f"Метки после преобразования: уникальные значения = {torch.unique(self.labels).tolist()}")

        # Определяем максимальную длину
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

        # Обрезаем если слишком длинная
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        # Создаем input_ids
        input_ids = seq.copy()
        attention_mask = [1] * len(input_ids)

        # Паддинг
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        # Получаем метку
        label = self.labels[idx]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': label
        }


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    Строит графики обучения
    """
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
    """
    Строит матрицу ошибок для многоклассовой классификации
    """
    # Конвертируем в numpy и уплощаем
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Проверяем длины
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

    # Вычисляем метрики для многоклассовой классификации
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"\nДетальные метрики:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

    # Выводим метрики для каждого класса
    print(f"\nМетрики по классам:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        prec = precision_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        rec = recall_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        f1 = f1_score(y_true, y_pred, labels=[i], average='macro') if i in np.unique(y_true) else 0
        print(f"{name:<10} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")


def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """
    Строит ROC-кривую
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC-кривая сохранена в {save_path}, AUC = {roc_auc:.4f}")


def plot_predictions_distribution(y_true, y_pred_probs, save_path='predictions_distribution.png'):
    """
    Строит распределение предсказанных вероятностей
    """
    plt.figure(figsize=(10, 6))

    # Разделяем по истинным классам
    true_0_probs = y_pred_probs[y_true == 0]
    true_1_probs = y_pred_probs[y_true == 1]

    plt.hist(true_0_probs, bins=30, alpha=0.7, label='Healthy (True)', color='green', density=True)
    plt.hist(true_1_probs, bins=30, alpha=0.7, label='Ascochyta (True)', color='red', density=True)

    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision boundary (0.5)')
    plt.xlabel('Predicted Probability of Ascochyta')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Распределение предсказаний сохранено в {save_path}")


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
        num_classes=4,  # для бинарной классификации
        save_path="ascochyta_classifier.pt",
        plots_dir="training_plots",
        device=None
):
    """
    Обучает классификатор аскохитоза
    """

    # Создаем директорию для графиков
    os.makedirs(plots_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загружаем датасет
    dataset = GenomicDataset(
        sequences_file=sequences_file,
        phenotypes_file=phenotypes_file,
        max_seq_len=None,
        pad_id=pad_id
    )

    # Разделяем на train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
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
        pad_id=pad_id
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

    # Функция потерь для бинарной классификации
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    # Для отслеживания лучшей модели и истории обучения
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
            attention_mask = batch['attention_mask'].to(device)
            #labels = batch['label'].to(device)
            labels = batch['label'].to(device).long()

            # Для бинарной классификации преобразуем метки в [B, 1]
            '''if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)'''

            if len(labels.shape) > 1:
                labels = labels.squeeze()

            # Forward
            embeddings = encoder(input_ids, attention_mask)
            logits = classifier(embeddings, attention_mask)

            # Loss
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
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

        # Validation
        encoder.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).long().squeeze()

                embeddings = encoder(input_ids, attention_mask)
                logits = classifier(embeddings, attention_mask)

                loss = criterion(logits, labels)

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)  # ← исправлено
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                val_all_labels.extend(labels.cpu().numpy())
                #val_all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Сохраняем лучшую модель
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

    # Строим графики обучения
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(plots_dir, 'training_history.png')
    )

    # Оцениваем на валидационной выборке с лучшей моделью
    print("\nОценка лучшей модели на валидационной выборке...")

    # Загружаем лучшую модель
    checkpoint = torch.load(save_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # После завершения обучения, при оценке лучшей модели
    encoder.eval()
    classifier.eval()

    val_all_labels = []
    val_all_preds = []  # ← собираем логиты вместо probs

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).long().squeeze()

            # Forward pass
            embeddings = encoder(input_ids, attention_mask)
            logits = classifier(embeddings, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            # Сохраняем результаты
            val_all_labels.extend(labels.cpu().numpy())
            val_all_preds.extend(predictions.cpu().numpy())

            # Отладочный вывод для первой итерации
            if batch_idx == 0:
                print(f"Debug - batch {batch_idx}:")
                print(f"  labels shape: {labels.shape}")
                print(f"  predictions shape: {predictions.shape}")
                print(f"  labels sample: {labels[:5].cpu().numpy()}")
                print(f"  predictions sample: {predictions[:5].cpu().numpy()}")

    print(f"\nСобрано {len(val_all_labels)} меток и {len(val_all_preds)} предсказаний")

    # Проверяем, что данные собраны
    if len(val_all_labels) == 0 or len(val_all_preds) == 0:
        print("ОШИБКА: Не удалось собрать предсказания!")
    else:
        # Конвертируем в numpy - ИСПРАВЛЕНО
        val_all_labels = np.array(val_all_labels).flatten()
        val_all_preds = np.array(val_all_preds).flatten()  # ← Исправлено: используем val_all_preds, а не val_all_probs

        print(f"Тип val_all_preds после конвертации: {type(val_all_preds)}")
        print(f"Shape val_all_preds: {val_all_preds.shape}")
        print(f"Уникальные метки: {np.unique(val_all_labels)}")
        print(f"Уникальные предсказания: {np.unique(val_all_preds)}")

        # Строим матрицу ошибок для 4 классов
        plot_confusion_matrix(
            val_all_labels, val_all_preds,
            class_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
            save_path=os.path.join(plots_dir, 'confusion_matrix.png')
        )

        # Выводим classification report для 4 классов
        print("\nClassification Report:")
        print(classification_report(
            val_all_labels,
            val_all_preds,
            target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4']
        ))

    # Строим матрицу ошибок
    '''plot_confusion_matrix(
        val_all_labels, val_all_preds,
        class_names=['Healthy', 'Ascochyta'],
        save_path=os.path.join(plots_dir, 'confusion_matrix.png')
    )'''

    # Строим ROC-кривую
    '''plot_roc_curve(
        val_all_labels, val_all_probs,
        save_path=os.path.join(plots_dir, 'roc_curve.png')
    )'''

    # Строим распределение предсказаний
    '''plot_predictions_distribution(
        val_all_labels, val_all_probs,
        save_path=os.path.join(plots_dir, 'predictions_distribution.png')
    )'''

    # Выводим classification report
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


def predict_ascochyta(model_path, sequences_file, phenotypes_file, plots_dir="prediction_plots", device=None):
    """
    Загружает модель и делает предсказания
    """
    os.makedirs(plots_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем сохраненную модель
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Создаем датасет для получения метаданных
    dataset = GenomicDataset(
        sequences_file=sequences_file,
        phenotypes_file=phenotypes_file,
        max_seq_len=config['max_seq_len'],
        pad_id=config['pad_id']
    )

    # Создаем модель с теми же параметрами
    encoder = GenomicTransformerEncoder(
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        pad_id=config['pad_id']
    )

    classifier = AscochytaClassifier(
        input_dim=config['proj_dim'],
        hidden_dim=128,
        num_classes=config['num_classes']
    )

    # Загружаем веса
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    encoder.eval()
    classifier.eval()

    # Делаем предсказания
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Making predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            embeddings = encoder(input_ids, attention_mask)
            logits = classifier(embeddings, attention_mask)
            probs = torch.sigmoid(logits)

            all_predictions.append(probs.cpu())
            all_labels.append(batch['label'].cpu())

    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Вычисляем метрики
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes.squeeze() == labels).float().mean().item()

    print(f"Accuracy on full dataset: {accuracy:.4f}")

    # Визуализация
    labels_np = labels.numpy().flatten()
    probs_np = predictions.numpy().flatten()
    preds_np = predicted_classes.numpy().flatten()

    # Матрица ошибок
    plot_confusion_matrix(
        labels_np, preds_np,
        class_names=['Healthy', 'Ascochyta'],
        save_path=os.path.join(plots_dir, 'final_confusion_matrix.png')
    )

    # ROC-кривая
    plot_roc_curve(
        labels_np, probs_np,
        save_path=os.path.join(plots_dir, 'final_roc_curve.png')
    )

    # Распределение
    plot_predictions_distribution(
        labels_np, probs_np,
        save_path=os.path.join(plots_dir, 'final_predictions_distribution.png')
    )

    return predictions, labels


# ============= ЗАПУСК ОБУЧЕНИЯ =============

if __name__ == "__main__":
    # Параметры
    SEQUENCES_FILE = "snp_batch_processing_results_20260312_115114/masked_sequences_ids.csv"
    PHENOTYPES_FILE = "pheno_ascoh.csv"  # файл с колонками SNP_ID и Ascochytosis
    VOCAB_SIZE = 12110
    PAD_ID = 0

    # Запускаем обучение классификатора
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
        num_classes=4,  # бинарная классификация
        save_path="ascochyta_transformer.pt",
        plots_dir="ascochyta_training_plots"
    )

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Лучшая точность на валидации: {history['best_val_acc']:.4f}")
    print(f"Графики сохранены в папке: ascochyta_training_plots/")
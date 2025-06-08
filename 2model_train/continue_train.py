import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import GlobalLocalNetwork
from embedding import ProteinEmbeddingGenerator
import warnings

warnings.filterwarnings("ignore")


class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class Trainer:
    def __init__(self, model_dir='protBERT', global_dim=1024, local_dim=1024,
                 num_classes=54, learning_rate=0.0001, save_dir='results'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Now using device:', self.device)
        self.embedding_generator = ProteinEmbeddingGenerator(model_dir, self.device)
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize model with reset_parameters
        self._init_model()

    def _init_model(self):
        """Initialize model with fresh weights"""
        self.model = GlobalLocalNetwork(
            self.global_dim, self.local_dim, self.num_classes
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def load_weights(self, epoch):
        """Load model weights from a specific epoch"""
        weights_path = os.path.join(self.save_dir, f'epoch_{epoch}_model.pth')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded weights from epoch {epoch}")
        else:
            print(f"Weights for epoch {epoch} not found. Starting from scratch.")

    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
            self.optimizer.zero_grad()

            global_emb, local_emb, mask = self.embedding_generator.embedding(sequences)
            global_emb = global_emb.to(self.device)
            local_emb = local_emb.to(self.device)
            mask = mask.to(self.device)
            labels = labels.long().to(self.device)


            with torch.cuda.amp.autocast():
                outputs = self.model(global_emb, local_emb, mask)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_preds)
        return total_loss / len(dataloader), metrics

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                global_emb, local_emb, mask = self.embedding_generator.embedding(sequences)
                global_emb = global_emb.to(self.device)
                local_emb = local_emb.to(self.device)
                mask = mask.to(self.device)
                labels = labels.long().to(self.device)


                with torch.cuda.amp.autocast():
                    outputs = self.model(global_emb, local_emb, mask)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_preds)
        return total_loss / len(dataloader), metrics, np.array(all_probs), np.array(all_labels)

    def _compute_metrics(self, y_true, y_pred):
        return {
            'acc': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }

    def _plot_metrics(self, train_metrics, val_metrics):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['loss'], label='Train')
        plt.plot(val_metrics['loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['acc'], label='Train')
        plt.plot(val_metrics['acc'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(os.path.join(self.save_dir, 'metrics.png'))
        plt.close()

    def train(self, csv_path, epochs=10, batch_size=32, start_epoch=1):
        df = pd.read_csv(csv_path)
        dataset = ProteinDataset(df['Sequence'].tolist(), df['Label'].tolist())

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []}
        val_metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []}

        if start_epoch > 1:
            self.load_weights(start_epoch - 1)

        try:
            for epoch in range(start_epoch, epochs + 1):
                # Training
                train_loss, train_met = self.train_step(train_loader)
                for k in train_metrics:
                    train_metrics[k].append(train_met[k] if k != 'loss' else train_loss)

                # Validation
                val_loss, val_met, _, _ = self.evaluate(val_loader)
                for k in val_metrics:
                    val_metrics[k].append(val_met[k] if k != 'loss' else val_loss)

                # Print epoch summary
                print(f"Epoch {epoch}/{epochs}")
                print(f"Train Loss: {train_loss:.4f} | Acc: {train_met['acc']:.4f} | Precision: {train_met['precision']:.4f} | Recall: {train_met['recall']:.4f} | F1: {train_met['f1']:.4f} | MCC: {train_met['mcc']:.4f}")
                print(f"Val Loss: {val_loss:.4f} | Acc: {val_met['acc']:.4f} | Precision: {val_met['precision']:.4f} | Recall: {val_met['recall']:.4f} | F1: {val_met['f1']:.4f} | MCC: {val_met['mcc']:.4f}")
                print("-" * 50)

                # Save model weights at the end of each epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f'epoch_{epoch}_model.pth')
                )
                print(f"Model weights saved for epoch {epoch} at {os.path.join(self.save_dir, f'epoch_{epoch}_model.pth')}")

            _, _, val_probs, val_labels = self.evaluate(val_loader)
            val_preds = np.argmax(val_probs, axis=1)

            # Store results
            all_y_true = val_labels
            all_y_pred = val_preds
            all_y_probs = val_probs

            # Plot metrics
            self._plot_metrics(train_metrics, val_metrics)

            # Generate final plots
            self._plot_confusion_matrix(all_y_true, all_y_pred)
            self._plot_roc_curve(all_y_true, all_y_probs)

        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress...")
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, 'interrupted_model.pth')
            )

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

    def _plot_roc_curve(self, y_true, y_score):
        # Compute macro ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= self.num_classes
        macro_auc = auc(all_fpr, mean_tpr)

        plt.figure()
        plt.plot(all_fpr, mean_tpr, color='b',
                 label=f'Macro ROC (AUC = {macro_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        # plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'))
        plt.close()


if __name__ == "__main__":
    trainer = Trainer(
        model_dir='../1protBERT',
        global_dim=1024,
        local_dim=1024,
        num_classes=54,
        learning_rate=0.0001,
        save_dir='../0data_save/model_weight'
    )
    trainer.train(
        csv_path='../0data_save/dataset.csv',
        epochs=2,
        batch_size=48,
        start_epoch=1  # start with certain epoch
    )

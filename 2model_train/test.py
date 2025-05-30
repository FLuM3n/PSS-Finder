import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)
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


class Tester:
    def __init__(self, model_dir='protBERT', global_dim=1024, local_dim=1024,
                 num_classes=54, save_dir='results'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('now using device:', self.device)
        self.embedding_generator = ProteinEmbeddingGenerator(model_dir, self.device)
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize model
        self.model = GlobalLocalNetwork(
            self.global_dim, self.local_dim, self.num_classes
        ).to(self.device)

    def load_model_weights(self, model_path):
        """Load model weights from a file"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Loaded model weights from {model_path}")

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

                outputs = self.model(global_emb, local_emb, mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

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

    def _save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix as a CSV file"""
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm_df.to_csv(os.path.join(self.save_dir, 'confusion_matrix.csv'))
        print(f"Confusion matrix saved to {os.path.join(self.save_dir, 'confusion_matrix.csv')}")

    def test(self, csv_path, model_path, batch_size=32):
        """Test the model on a test set"""
        # Load model weights
        self.load_model_weights(model_path)

        # Load test data
        df = pd.read_csv(csv_path)
        test_dataset = ProteinDataset(df['Sequence'].tolist(), df['Label'].tolist())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate on test set
        test_loss, test_metrics, test_probs, test_labels = self.evaluate(test_loader)

        # Print test metrics
        print("\nTest Metrics:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_metrics['acc']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"MCC: {test_metrics['mcc']:.4f}")

        # Save confusion matrix as CSV
        self._save_confusion_matrix(test_labels, np.argmax(test_probs, axis=1))


if __name__ == "__main__":
    tester = Tester(
        model_dir='../1protBERT',
        global_dim=1024,
        local_dim=1024,
        num_classes=54,
        save_dir='../0data_save/49epoch_test'
    )

    # Test the model on the test set
    tester.test(
        csv_path='../0data_save/test.csv',  # Path to test dataset
        model_path='../0data_save/model_weight/epoch_49_model.pth',  # Path to trained model weights
        batch_size=16
    )
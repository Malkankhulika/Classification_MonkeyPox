import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2
from Utils.getData import Data
from torch.utils.data import DataLoader

# Fungsi untuk evaluasi model dan menampilkan metrik evaluasi
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for src, trg in test_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            preds = model(src)
            preds = torch.argmax(preds, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(trg.cpu().numpy())

    # Calculate evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return acc, prec, rec, f1, all_labels, all_preds

# Fungsi untuk menampilkan dan menyimpan confusion matrix
def plot_confusion_matrix(all_labels, all_preds, num_classes, save_path="confusion_matrix.png"):
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Purples", fmt=".2f", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title("Confusion Matrix (Normalized)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(save_path)
    plt.show()

# Fungsi utama untuk menjalankan evaluasi
def main():
    # Parameter
    BATCH_SIZE = 32
    NUM_CLASSES = 6
    model_path = "trained_mobilenet_v2.pth"

    # Load Dataset
    aug_path = "C:/LISIKASI/DATASET/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/LISIKASI/DATASET/Original Images/Original Images/FOLDS/"
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluasi Model
    acc, prec, rec, f1, all_labels, all_preds = evaluate_model(model, test_loader)
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot Confusion Matrix
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]
    plot_confusion_matrix(all_labels, all_preds, NUM_CLASSES)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18  # Import model ResNet dari torchvision
from Utils.getData import Data

def main():
    # Hyperparameter
    BATCH_SIZE = 64
    EPOCH = 18
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6

    # Path ke dataset
    aug_path = "C:/LISIKASI/DATASET/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/LISIKASI/DATASET/Original Images/Original Images/FOLDS/"

    # Load data
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    train_data = dataset.dataset_train + dataset.dataset_aug
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Inisialisasi model ResNet
    model = resnet18(pretrained=True)  # Menggunakan ResNet18 dengan bobot pre-trained
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Menyesuaikan output layer dengan jumlah kelas

    # Inisialisasi loss function dan optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training
    train_losses = train_model(train_loader, model, loss_fn, optimizer, EPOCH)

    # Simpan model
    torch.save(model.state_dict(), "trained_resnet18.pth")

    # Visualisasi loss
    plt.plot(range(EPOCH), train_losses, color="#ff69b4", label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training_resnet.png")

def train_model(train_loader, model, loss_fn, optimizer, epochs):
    """
    Fungsi untuk melatih model.
    :param train_loader: DataLoader untuk data latih
    :param model: Model yang akan dilatih
    :param loss_fn: Fungsi loss
    :param optimizer: Optimizer
    :param epochs: Jumlah epoch
    :return: Daftar nilai loss per epoch
    """
    train_losses = []

    for epoch in range(epochs):
        model.train()
        loss_train = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            # Mengubah dimensi data menjadi format PyTorch (C, H, W)
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            # Forward pass
            pred = model(src)
            loss = loss_fn(pred, trg)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Menghitung statistik
            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss_train / len(train_loader):.4f}, Accuracy: {accuracy_train:.2f}%")
        train_losses.append(loss_train / len(train_loader))

    return train_losses

if __name__ == "__main__":
    main()
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.dataset import BrailleCellDataset
from cnn.model import BrailleCNN


def train_model(
    image_paths,
    labels,
    num_epochs=10,
    batch_size=32,
    lr=0.001,
    device=None,
    save_path="braille_cnn.pth",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] range
        ]
    )

    dataset = BrailleCellDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BrailleCNN(num_classes=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
        )

    # Save the trained model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

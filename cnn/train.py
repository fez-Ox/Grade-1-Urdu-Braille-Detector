import glob
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.dataset import BrailleCellDataset
from cnn.model import BrailleCNN
from preprocessing.pipeline import preprocess_img
from segmentation.cropper import crop_and_save_cells
from segmentation.dot_detection import detect_dots
from segmentation.grouping import group_into_lines, segment_cells_from_lines


def train_model(
    image_paths,
    labels,
    num_epochs=10,
    batch_size=32,
    lr=0.001,
    device=None,
    save_path="braille_cnn.pth",
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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


def dot_string_to_class_int(dot_str: str) -> int:
    """
    Converts a standard Braille dot string (like '134', '1', or '0' for space)
    into our 0-63 class integer.
    """
    if dot_str == "0":
        return 0

    class_val = 0
    for char in dot_str:
        dot_num = int(char)
        # Dot 1 maps to bit 0, Dot 2 to bit 1, etc.
        class_val |= 1 << (dot_num - 1)

    return class_val


if __name__ == "__main__":
    images_dir = "inputs/images"
    dots_dir = "inputs/dots"
    output_crops_dir = "cnn_dataset/crops"

    print("--- 1. Generating Training Crops ---")

    if os.path.exists(output_crops_dir):
        shutil.rmtree(output_crops_dir)
    os.makedirs(output_crops_dir, exist_ok=True)

    temp_dip_dir = "cnn_dataset/temp_dip"
    os.makedirs(os.path.join(temp_dip_dir, "cleaned"), exist_ok=True)
    os.makedirs(os.path.join(temp_dip_dir, "binary"), exist_ok=True)

    image_paths = []
    labels = []

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    # Use a subset of images for testing if needed, or all of them.
    max_images = 100
    image_files = image_files[:max_images]

    successful_images = 0

    for idx, img_path in enumerate(image_files):
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]

        dots_path = os.path.join(dots_dir, f"{img_id}.txt")
        if not os.path.exists(dots_path):
            continue

        with open(dots_path, "r") as f:
            gt_text = f.read().strip()

        if not gt_text:
            continue

        # Ignore zeros (spaces) as DIP only segments visible physical cells
        gt_dot_strings = [d for d in gt_text.split() if d != "0"]

        try:
            _, _, img_binary = preprocess_img(img_path, temp_dip_dir)
            detected_dots = detect_dots(img_binary)
            lines = group_into_lines(detected_dots, eps=15.0)
            cells = segment_cells_from_lines(
                lines, col_group_eps=10.0, intra_cell_threshold=17.0
            )
        except Exception:
            continue

        if len(cells) == len(gt_dot_strings):
            successful_images += 1
            crop_and_save_cells(img_binary, cells, temp_dip_dir)

            for i, cell in enumerate(cells):
                temp_crop_path = os.path.join(
                    temp_dip_dir, f"cell_{cell.order_index:03d}.png"
                )
                final_crop_path = os.path.join(
                    output_crops_dir, f"{img_id}_cell_{i:03d}.png"
                )

                os.rename(temp_crop_path, final_crop_path)
                class_int = dot_string_to_class_int(gt_dot_strings[i])

                image_paths.append(final_crop_path)
                labels.append(class_int)

        if (idx + 1) % 20 == 0:
            print(
                f"Processed {idx + 1}/{len(image_files)} images. Extracted cells: {len(image_paths)}"
            )

    print(
        f"\nExtracted {len(image_paths)} perfect Braille cell crops from {successful_images} images."
    )

    if len(image_paths) > 0:
        print("\n--- 2. Starting CNN Training ---")
        train_model(
            image_paths=image_paths,
            labels=labels,
            num_epochs=15,
            batch_size=64,
            lr=0.001,
            save_path="braille_cnn_final.pth",
        )
        print("\nPipeline Complete. Model saved to 'braille_cnn_final.pth'")
    else:
        print("No perfect crops generated. Cannot train.")

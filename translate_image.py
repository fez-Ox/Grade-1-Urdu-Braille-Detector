import os
import sys

# Ensure Notebook environments find local modules when running this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import louis
import torch
from PIL import Image
from torchvision import transforms

from cnn.mappings import int_to_unicode_braille
from cnn.model import BrailleCNN
from preprocessing.pipeline import preprocess_img
from segmentation.cropper import crop_and_save_cells
from segmentation.dot_detection import detect_dots
from segmentation.grouping import group_into_lines, segment_cells_from_lines


def load_model(weights_path="braille_cnn_final.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrailleCNN(num_classes=64)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, device


def predict_crop(model, device, transform, img_crop):
    # Convert OpenCV numpy image (grayscale) to PIL Image
    pil_img = Image.fromarray(img_crop).convert("L")
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        _, predicted = outputs.max(1)
        class_idx = predicted.item()

    return int_to_unicode_braille(class_idx)


def translate_image(
    image_path, model_path="braille_cnn_final.pth", table_path="./tables/ur-pk-g2.ctb"
):
    print(f"Translating image: {image_path}")

    # 1. Load model
    model, device = load_model(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 2. DIP Segmentation
    temp_dir = "cnn_dataset/temp_dip"
    os.makedirs(os.path.join(temp_dir, "cleaned"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "binary"), exist_ok=True)

    _, _, img_binary = preprocess_img(image_path, temp_dir)
    dots = detect_dots(img_binary)
    # Increased eps from 15.0 to 25.0 to prevent a single wandering cell from forming its own line
    lines = group_into_lines(dots, eps=25.0)
    cells = segment_cells_from_lines(
        lines, col_group_eps=10.0, intra_cell_threshold=17.0
    )

    # Crop the cells (populates cell.image)
    crop_and_save_cells(img_binary, cells, temp_dir)

    # 3. Predict & Reconstruct Spacing
    braille_unicode = ""

    # Group cells by line index to handle spacing correctly
    lines_of_cells = {}
    for cell in cells:
        lines_of_cells.setdefault(cell.line_index, []).append(cell)

    for line_idx in sorted(lines_of_cells.keys()):
        line_cells = sorted(lines_of_cells[line_idx], key=lambda c: c.bbox[0])

        for i, cell in enumerate(line_cells):
            # Predict character
            char = predict_crop(model, device, transform, cell.image)
            braille_unicode += char

            # Check gap to next cell to insert spaces
            if i < len(line_cells) - 1:
                next_cell = line_cells[i + 1]
                # Compare X coordinates of bounding boxes.
                # Inter-cell gap is ~21px. Word gap is ~56px.
                dist = next_cell.bbox[0] - cell.bbox[0]
                if dist > 45:  # restored to 45 to safely catch spaces
                    braille_unicode += chr(0x2800)  # Unicode Braille Space

        braille_unicode += "\n"

    print("\n--- Extracted Braille ---")
    print(braille_unicode.strip())

    # 4. Translate via LibLouis (backTranslateString converts Braille to Text)
    print("\n--- Translated Urdu ---")
    urdu_translation = louis.backTranslateString([table_path], braille_unicode)
    print(urdu_translation.strip())

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/urdu_translation.txt", "w") as f:
        f.write(urdu_translation.strip())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "inputs/images/chunk_00004.jpg"

    translate_image(img_path)

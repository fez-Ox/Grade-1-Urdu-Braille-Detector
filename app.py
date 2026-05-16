import os
import sys
import matplotlib
matplotlib.use('Agg')

import torch
import gradio as gr
import numpy as np
import louis
from PIL import Image
from torchvision import transforms

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn.mappings import int_to_unicode_braille
from cnn.model import BrailleCNN
from preprocessing.pipeline import preprocess_img
from segmentation.cropper import crop_and_save_cells
from segmentation.dot_detection import detect_dots
from segmentation.grouping import group_into_lines, segment_cells_from_lines
from visualization.storyboard import create_storyboard

# --- Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = BrailleCNN(num_classes=64)
WEIGHTS_PATH = "braille_cnn_final.pth"

if os.path.exists(WEIGHTS_PATH):
    MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE)
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

TABLE_PATH = "./tables/ur-pk-g2.ctb"

def predict_crop(img_crop):
    pil_img = Image.fromarray(img_crop).convert("L")
    tensor_img = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(tensor_img)
        _, predicted = outputs.max(1)
        class_idx = predicted.item()
    return int_to_unicode_braille(class_idx)

def process_braille(image_input):
    if image_input is None:
        return "Please upload an image.", None
    
    # Save temp input
    temp_input = "temp_gradio_input.png"
    # Gradio provides numpy array (RGB)
    Image.fromarray(image_input).save(temp_input)
    
    temp_dir = "outputs/gradio_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. DIP Preprocessing
    img_bgr, img_gray, img_binary = preprocess_img(temp_input, temp_dir)
    
    # 2. Segmentation
    dots = detect_dots(img_binary)
    lines = group_into_lines(dots, eps=25.0)
    cells = segment_cells_from_lines(lines, col_group_eps=10.0, intra_cell_threshold=17.0)
    
    # Crop cells
    crop_and_save_cells(img_binary, cells, temp_dir)
    
    # 3. Prediction & Translation
    braille_unicode = ""
    lines_of_cells = {}
    for cell in cells:
        lines_of_cells.setdefault(cell.line_index, []).append(cell)
        
    for line_idx in sorted(lines_of_cells.keys()):
        line_cells = sorted(lines_of_cells[line_idx], key=lambda c: c.bbox[0])
        for i, cell in enumerate(line_cells):
            char = predict_crop(cell.image)
            braille_unicode += char
            
            if i < len(line_cells) - 1:
                next_cell = line_cells[i+1]
                dist = next_cell.bbox[0] - cell.bbox[0]
                if dist > 45:
                    braille_unicode += chr(0x2800) # Space
        braille_unicode += "\n"

    urdu_translation = louis.backTranslateString([TABLE_PATH], braille_unicode)
    
    # 4. Generate Storyboard
    storyboard_path = create_storyboard(
        img_bgr, img_gray, img_binary, dots, lines, cells, 
        output_path=os.path.join(temp_dir, "storyboard.png")
    )
    
    return urdu_translation.strip(), braille_unicode.strip(), storyboard_path

# --- Gradio UI ---
with gr.Blocks(title="Urdu Braille OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Urdu Braille OCR System")
    gr.Markdown("### Powered by Custom Digital Image Processing (diplib.py) & CNN")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Braille Image", type="numpy")
            btn = gr.Button("Translate Braille", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Translated Urdu Text", lines=3, text_align="right", interactive=False)
            output_braille = gr.Textbox(label="Detected Braille Unicode", lines=3, interactive=False)

    gr.Markdown("## Pipeline Storyboard (DIP Stages)")
    storyboard_img = gr.Image(label="Pipeline Overview", interactive=False)
    
    btn.click(
        fn=process_braille,
        inputs=input_img,
        outputs=[output_text, output_braille, storyboard_img]
    )
    
    gr.Examples(
        examples=["chunk_111141.jpg", "chunk_99993.jpg"],
        inputs=input_img
    )

if __name__ == "__main__":
    demo.launch(share=True)

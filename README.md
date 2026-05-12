# Urdu Braille OCR System

A complete pipeline for detecting and translating Urdu Braille from images using deep learning and digital image processing.

## Overview

This project implements an OCR system for Urdu Braille that consists of three main components:

1. **DIP Segmentation** - Uses Digital Image Processing techniques to detect and segment Braille cells from images
2. **CNN Classification** - A Convolutional Neural Network to classify individual Braille cell patterns
3. **Translation** - Uses LibLouis library to convert Braille to Urdu text

## Project Structure

```
.
├── cnn/                    # CNN model and training
│   ├── model.py            # BrailleCNN architecture
│   ├── train.py            # Training script
│   ├── dataset.py          # Dataset class
│   └── mappings.py        # Braille to Unicode mappings
├── preprocessing/          # Image preprocessing
│   └── pipeline.py        # DIP preprocessing pipeline
├── segmentation/          # Braille cell segmentation
│   ├── dot_detection.py   # Detect Braille dots
│   ├── grouping.py        # Group dots into lines/cells
│   └── cropper.py         # Crop detected cells
├── translate_image.py      # Main inference script
├── requirements.txt        # Python dependencies
├── tables/                 # LibLouis translation tables
└── inputs/                 # Input images and ground truth
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### System Dependencies (Linux)

For Braille translation with LibLouis:

```bash
apt-get update
apt-get install -y liblouis-dev python3-louis
```

## Dataset

The project uses the [Braille Urdu](https://www.kaggle.com/datasets/cookiefinder/braille-urdu) dataset from Kaggle.

The dataset should be placed as:
```
inputs/
├── images/    # Input Braille images (.jpg)
└── dots/      # Ground truth Braille dot patterns (.txt)
```

## Training the Model

The easiest way to train the model is using the provided Jupyter notebook:

### Using Google Colab (Recommended)

1. Open `DIP_Model_Training.ipynb` in Google Colab
2. Run the cells sequentially

The notebook performs:
1. Downloads the Braille Urdu dataset from Kaggle
2. Installs system dependencies (liblouis)
3. Sets up the repository
4. Trains the CNN model (15 epochs)
5. Saves the model to `braille_cnn_final.pth`

### Local Training

If you have the dataset locally:

```bash
python cnn/train.py
```

This will:
1. Extract Braille cell crops from input images using DIP
2. Train the CNN for 15 epochs (configurable)
3. Save the model to `braille_cnn_final.pth`

### Training Parameters

Modify these in `cnn/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_epochs | 15 | Number of training epochs |
| batch_size | 64 | Batch size |
| learning_rate | 0.001 | Adam optimizer learning rate |
| max_images | 2000 | Maximum images to use for training |

## Running Inference

Translate a Braille image to Urdu text:

```bash
python translate_image.py path/to/image.jpg
```

The script will:
1. Preprocess the image (denoising, binarization)
2. Detect and segment Braille cells
3. Classify each cell using the CNN
4. Reconstruct spacing (words/sentences)
5. Translate Braille to Urdu via LibLouis
6. Save output to `outputs/urdu_translation.txt`

### Default Image

If no image path is provided, it defaults to `inputs/images/chunk_00004.jpg`.

## Model Architecture

The CNN (`cnn/model.py`) is a lightweight classifier:

- Input: 64x64 grayscale image
- 3 convolutional blocks (16 → 32 → 64 channels)
- MaxPooling after each block
- Fully connected layers (4096 → 256 → 64)
- Output: 64 classes (representing all Braille dot combinations)

Training achieves ~97.6% accuracy on the validation set.

## Output

The system outputs:
- Extracted Unicode Braille characters
- Translated Urdu text

## License

This project is based on the [Grade-1-Urdu-Braille-Detector](https://github.com/fez-Ox/Grade-1-Urdu-Braille-Detector) repository.

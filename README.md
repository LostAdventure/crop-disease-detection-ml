# Crop Disease Detection and Prediction System

**Project:** Development of a Crop Disease Detection and Prediction System Using Supervised Machine Learning Algorithms

**Developers**
1. EZRAH KIPROTICH KOSGEI — SEZRKO2411
2. RICHARD WAMALWA MAYU — SWAMMA2411
3. BRIGHTON NABISWA — SBRINA2411

## Overview
This project trains a supervised Convolutional Neural Network (CNN) to classify crop leaf images into the labelled healthy/disease categories in the PlantVillage dataset. It also provides a simple Streamlit interface for testing a trained model with an uploaded image.

## Dataset
The training script uses the PlantVillage dataset through TensorFlow Datasets. It downloads the dataset on first run. The script creates deterministic splits:
- 70% training
- 15% validation
- 15% testing

The dataset is not included in this GitHub repository. Dataset availability and terms should be checked at the source before reuse.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection for the initial dataset download
- At least 8 GB RAM recommended; a GPU or Google Colab runtime can speed up training

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train and evaluate
From the project root:
```bash
python src/train.py --epochs 20 --batch-size 32
```

The script saves the best model and evaluation outputs in `artifacts/`:
- `crop_disease_model.keras` — trained model
- `class_names.json` — class labels
- `metrics.json` — test accuracy and macro/weighted precision, recall and F1
- `classification_report.json` — per-class evaluation
- `confusion_matrix.png` — confusion matrix

**Important:** Do not write performance scores in the project report until this training script has completed. Copy the actual test metrics from `artifacts/metrics.json`.

## Run the web application
After training:
```bash
streamlit run app.py
```
Open the local address printed by Streamlit, then upload a JPG or PNG leaf image.

## Method and evaluation
- Category: supervised multi-class image classification
- Algorithm: CNN with convolution, batch normalization, pooling, dropout and softmax output
- Preprocessing: resize to 224 × 224 pixels and normalize pixel values to [0, 1]
- Training augmentation: random horizontal flip, rotation and zoom
- Validation: held-out validation split and early stopping
- Testing: independent held-out test split
- Metrics: accuracy, macro precision, macro recall, macro F1-score, weighted metrics and confusion matrix

## Limitations
PlantVillage images were largely captured in controlled settings. A strong score on this dataset does not guarantee equally strong performance on photographs from farms with different lighting, backgrounds, camera quality or local crop varieties. Test with representative field images and seek agricultural expert confirmation before recommending treatment.

## Suggested future improvements
1. Collect and label more Kenyan field images across seasons and regions.
2. Compare the CNN with transfer-learning models such as MobileNetV2 or EfficientNet.
3. Add disease-management guidance reviewed by agricultural experts.
4. Optimize the model for offline Android use.
5. Evaluate class imbalance and report per-class recall, not accuracy alone.

## References
- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics through machine learning and crowdsourcing.
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, 1419. https://doi.org/10.3389/fpls.2016.01419
- TensorFlow Datasets PlantVillage: https://www.tensorflow.org/datasets/catalog/plant_village

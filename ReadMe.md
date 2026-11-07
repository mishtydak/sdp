# 🏪 KiranaShopDemo – AI-Powered Visual Inventory Assessment

An AI-powered computer vision prototype that analyzes images of kirana (small retail) shops to detect products, estimate visible inventory quantities, and calculate approximate inventory value.

The project uses two independently trained Faster R-CNN models with ResNet-50 and Feature Pyramid Networks (FPN) to identify standard packaged products and additional kirana-specific objects.

---

## 📌 Problem Statement

Small retail shops often manage their inventory manually. Tracking product quantities and estimating inventory value can be time-consuming, especially when products are arranged across multiple shelves.

KiranaShopDemo explores how computer vision can assist shop owners by analyzing shop images, identifying visible products, estimating quantities, and calculating approximate inventory value.

## 🎯 Project Objectives

- Detect multiple product categories from shop images.
- Identify additional kirana-specific objects such as hanging sachet strips and loose grains.
- Estimate visible product quantities using object detection.
- Calculate approximate inventory value using editable product prices.
- Provide annotated images and category-wise inventory summaries.

## ✨ Key Features

- **Multi-image Upload:** Upload multiple shop images for assessment.
- **Product Detection:** Detect 65 standard packaged product categories.
- **Custom Object Detection:** Identify hanging sachet strips and loose grains using a separately trained model.
- **Visual Annotation:** Display bounding boxes around detected objects.
- **Inventory Counting:** Calculate category-wise detected quantities.
- **Editable Price Catalogue:** Customize unit prices for different product categories.
- **Value Estimation:** Calculate approximate value using detected quantity × unit price.
- **CSV Export:** Download inventory assessment results.

## 🧠 Model Architecture

The system uses two separate object detection models.

### 1. Standard Product Detection

**Faster R-CNN with ResNet-50-FPN**

- Backbone: ResNet-50
- Feature extraction: Feature Pyramid Network (FPN)
- Detection architecture: Faster R-CNN
- Number of product categories: 65
- Number of model classes: 66 (including background)

### 2. Custom Object Detection

**Faster R-CNN with ResNet-50-FPN**

Trained separately to recognize two additional categories:

- `hanging_sachet_strip`
- `loose_grains`

The model has three output classes, including background.

### Overall Workflow

1. User uploads shop images.
2. Images are preprocessed and converted into tensors.
3. Both detection models perform inference.
4. Predictions are filtered using a confidence threshold.
5. Detection results from both models are combined.
6. Products are grouped by category and quantities are calculated.
7. Estimated values are calculated using the editable price catalogue.
8. Results are displayed through the Streamlit interface.

## 📊 Model Evaluation

The following metrics were obtained from the held-out test set of the original 65-class product detection model.

| Evaluation Metric | Result |
|---|---:|
| mAP @ IoU 0.50:0.95 | 57.57% |
| mAP @ IoU 0.50 | 76.15% |
| mAP @ IoU 0.75 | 69.14% |
| AP Small Objects | 17.12% |
| AP Medium Objects | 46.35% |
| AP Large Objects | 66.57% |
| Average Recall (100 detections) | 69.70% |

### Counting Evaluation

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 5.27 |
| Root Mean Squared Error (RMSE) | 10.77 |
| Exact Count Accuracy | 21.80% |
| Within 2 Items Accuracy | 57.14% |

**Note:** These evaluation results belong to the original 65-class model only. Numerical evaluation metrics for the custom two-class model have not yet been established.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Object Detection | Faster R-CNN |
| Backbone | ResNet-50 |
| Feature Extraction | Feature Pyramid Network |
| Web Application | Streamlit |
| Data Handling | Pandas |
| Model Evaluation | COCO evaluation metrics |

## 📁 Project Structure

```text
KiranaShopDemo/
│
├── custom_detector/
│   └── kirana_custom_fasterrcnn.pth
│
├── app.py
├── categories.json
├── fasterrcnn_grouped_best.pth
├── requirements.txt
├── .gitignore
└── README.md
```

**Model checkpoints are excluded from this Git repository because of their large file sizes.** They must be downloaded separately and placed in the expected directories before running the application.

Expected checkpoint locations:

- `fasterrcnn_grouped_best.pth` – project root.
- `custom_detector/kirana_custom_fasterrcnn.pth` – custom detector directory.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- pip.
- Model checkpoint files.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/KiranaShopDemo.git
cd KiranaShopDemo
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch and Torchvision versions for your system if they are not already specified in `requirements.txt`.

### 4. Add model checkpoints

Download the trained model files and place them in the expected locations described in the Project Structure section.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.


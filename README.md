# Multimodal Deep Learning Model for Diabetes Hyperglycemic Prediction

A system developed as part of the Deep Learning course at the University of Salerno, focused on predicting hyperglycemic episodes using multimodal data integration.
The solution is based on the HUPA-UCM Diabetes Dataset and employs a multimodal deep learning architecture that combines Continuous Glucose Monitoring (CGM), Fitbit wearable data, and insulin therapy and carbohydrates information.
The model integrates heterogeneous data sources through specialized LSTM branches with attention mechanisms and cross-modal fusion layers, evaluated through standard classification metrics such as Accuracy, Precision, Recall, and AUC.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/cirovitale/diabetes-multimodal
cd diabetes-multimodal
```

### 2. Environment Setup

#### Create and Activate Virtual Environment

```bash
python -m venv venv

# Activate the environment
venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

1. Download the HUPA-UCM dataset from: [https://data.mendeley.com/datasets/3hbcscwz44/1](https://data.mendeley.com/datasets/3hbcscwz44/1)
2. Extract the downloaded files
3. Place the following folders in the `/dataset` directory:
   - `Preprocessed/`: Contains preprocessed patient data files (HUPA\*.csv)
   - `Raw_Data/`: Contains original raw data files

## Usage

### Training and Evaluation

```bash
python main.py
```

This will:

- Load and preprocess the HUPA-UCM dataset
- Train the multimodal model and baseline models
- Generate evaluation metrics and comparison plots
- Save the best performing model

### Model Loading and Inference

```python
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('models/best_diabetes_model.h5')

# Use for inference on new data
predictions = model.predict([cgm_data, fitbit_data, insulin_data])
```

## Project Structure

```
diabetes-multimodal/
├── main.py                     # Main training and evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
├── dataset/                    # HUPA-UCM dataset
│   ├── Preprocessed/
│   └── Raw_Data/
├── models/
│   └── best_diabetes_model.h5  # Best multimodal model
├── plots/                      # Generated visualizations
│   ├── data_distribution_preprocessed.png
│   ├── data_distribution_raw.png
│   ├── multimodal_model_architecture.png
│   └── training_history.png
└── Documentazione.pdf      # Documentation (Italian)
```

## Documentation

The complete documentation of the project, including methodology, dataset analysis, model architecture and results, is available in Italian language in: [Documentazione.pdf](https://github.com/cirovitale/diabetes-multimodal/blob/main/Documentazione.pdf)

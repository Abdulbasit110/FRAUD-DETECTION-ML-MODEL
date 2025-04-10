# Fraud Detection Pipeline

A machine learning pipeline for detecting suspicious financial transactions.

## Overview

This pipeline processes raw transaction data through a series of steps to build a fraud detection model:

1. **Data loading**: Imports raw transaction data from CSV files
2. **Data cleaning**: Handles missing values and converts data types
3. **Feature engineering**: Creates transaction pattern metrics and risk indicators
4. **Model training**: Trains various classifiers (Random Forest, SVM, etc.)
5. **Evaluation**: Measures model performance with standard metrics

## Features

- **Transaction Analysis**: Detects unusual transaction patterns and spiking
- **Email Matching**: Identifies discrepancies between sender names and email domains
- **Multiple Classifiers**: Random Forest, Logistic Regression, SVM, KNN, and more
- **Class Imbalance Handling**: Uses SMOTE to address the rarity of fraudulent transactions
- **Performance Visualization**: Generates confusion matrices and performance graphs

## Installation

```bash
# Clone the repository
git clone https://github.com/Abdulbasit110/FRAUD-DETECTION-ML-MODEL.git
cd fraud-detection-pipeline

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from fraud_detection_pipeline import FraudDetectionPipeline

# Initialize the pipeline
pipeline = FraudDetectionPipeline(
    raw_data_path="path/to/your/data.csv",
    model_save_path="random_forest_model.pkl"
)

# Run the complete pipeline
results = pipeline.run_pipeline(model_type='random_forest')

# Save the trained model
pipeline.save_model()

# Make predictions on new data
new_transactions = pd.read_csv("new_transactions.csv")
predictions, probabilities = pipeline.predict_transaction(new_transactions)
```

### Advanced Usage

You can run each step of the pipeline individually:

```python
pipeline = FraudDetectionPipeline()

# Load and prepare data
pipeline.load_data("path/to/your/data.csv")
pipeline.clean_data()
pipeline.engineer_features()
pipeline.prepare_data_for_training(test_size=0.25, random_state=42)

# Train different models and compare
models = ['random_forest', 'logistic_regression', 'svm', 'voting']
results = {}

for model_type in models:
    pipeline.train_model(model_type)
    results[model_type] = pipeline.evaluate_model()
    
# Save the best model
best_model = max(results, key=lambda k: results[k]['f1'])
pipeline.train_model(best_model)
pipeline.save_model(f"{best_model}_model.pkl")
```

## Model Performance

The pipeline includes evaluation metrics for model performance:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

A confusion matrix visualization is saved as `confusion_matrix.png` after evaluation.

## Customization

You can customize the pipeline by:
- Adding new features in the `engineer_features()` method
- Implementing different models in the `train_model()` method
- Modifying the evaluation metrics in `evaluate_model()`

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- Flask (for API integration)
- joblib


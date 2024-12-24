
---

# Fraud Detection Using Machine Learning

## Overview
This project leverages machine learning to build a fraud detection system that identifies suspicious transactions. It includes data preprocessing, feature engineering, and model evaluation to detect anomalies with high accuracy. The system utilizes advanced classification techniques and performance metrics to ensure robustness and precision.

![system architecture](<system architecture.png>)
---

## Key Features
- **Data Processing**: Handles missing data, outliers, and time-series patterns for transaction-level analysis.
- **Metrics Calculation**: Includes transaction volume, frequency patterns, and sender-beneficiary relationships.
- **Fraud Detection**: Classifies transactions into 'Genuine' or 'Suspicious' using a combination of machine learning classifiers.
- **Model Evaluation**: Measures performance with accuracy, precision, recall, and F1-score metrics.
- **Visualization**: Includes confusion matrices and performance graphs for detailed insights.

---

## Classifiers Used
- Logistic Regression
- Support Vector Classifier
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Gaussian Naive Bayes
- Ridge Classifier
- Voting Classifier (Ensemble)

---

## Tools and Libraries
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **scikit-learn**: Machine learning models and metrics
- **SMOTE**: For handling class imbalance
- **Matplotlib & Seaborn**: Visualization
- **imblearn**: Over-sampling techniques

---

## Data Insights
- **Spiking Detection**: Identifies unusual transaction spikes over specific durations.
- **Email Matching**: Matches sender names with beneficiary email patterns for potential fraud indicators.
- **Transaction Patterns**: Analyzes top and bottom transaction volumes and frequencies.

---

## Performance
- **Training and Testing Accuracy**: Ensures high precision with a threshold of 90% (visualized in the attached accuracy report).
- **Cross-Validation**: Validates model robustness across different splits.
- **Confusion Matrices**: Visual representation of prediction accuracy (e.g., 'Genuine' vs. 'Suspicious').
![alt text](<accuracy report.png>)

![alt text](<Screenshot 2024-12-15 113416.png>)

![alt text](<Screenshot 2024-12-15 113547.png>)

![alt text](<Screenshot 2024-12-15 113644.png>)

![alt text](<Screenshot 2024-12-15 113753.png>)

![alt text](<Screenshot 2024-12-15 113832.png>)
---

## Instructions to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/abdulbasit110/FRAUD-DETECTION-ML-MODEL.git
   cd SW Project Final.ipynb
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Place your dataset in the project folder and update the file path in the code.

4. **Run the Model**
   Execute the script or notebook to preprocess the data, train models, and evaluate performance.

---

## Visualization
Performance metrics and classifier comparison are visualized using confusion matrices and accuracy plots. Example plots are attached in the project (e.g., `accuracy_report.png`).

---

## Future Enhancements
- Incorporating deep learning models for enhanced fraud detection.
- Implementing real-time transaction analysis.
- Expanding feature engineering for more sophisticated fraud patterns.

---

## Contact
For questions or collaboration opportunities, reach out at:  
**Email**: abdulbasit4408944@gmail.com  
**GitHub**: [abdulbasit110](https://github.com/abdulbasit110)


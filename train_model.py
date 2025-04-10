from fraud_detection_pipeline import FraudDetectionPipeline
import os

def main():
    # Check if data file exists
    data_path = "project_dataset_sw/project_dataset_sw.csv"
    if not os.path.exists(data_path):
        # Look for the data in the parent folder
        alt_data_path = "../project_dataset_sw/project_dataset_sw.csv"
        if os.path.exists(alt_data_path):
            data_path = alt_data_path
        else:
            print(f"Error: Data file not found at {data_path} or {alt_data_path}")
            return

    print(f"Using data from: {data_path}")
    
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline(
        raw_data_path=data_path,
        model_save_path="random_forest_model.pkl"
    )
    
    # Run the pipeline steps individually for better control
    pipeline.load_data()
    pipeline.clean_data()
    pipeline.engineer_features()
    
    # Use smaller subset for quicker training (adjust as needed)
    pipeline.prepare_data_for_training(test_size=0.2)
    
    # Train a simple Random Forest model
    pipeline.train_model(model_type='random_forest')
    
    # Evaluate the model
    metrics = pipeline.evaluate_model()
    print("Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save the trained model
    pipeline.save_model()
    print(f"Model saved to {pipeline.model_save_path}")

if __name__ == "__main__":
    main() 
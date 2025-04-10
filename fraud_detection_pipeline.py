import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionPipeline:
    def __init__(self, raw_data_path=None, model_save_path='random_forest_model.pkl'):
        """
        Initialize the fraud detection pipeline.
        
        Parameters:
        -----------
        raw_data_path : str
            Path to the raw transaction data CSV file
        model_save_path : str
            Path where the trained model will be saved
        """
        self.raw_data_path = raw_data_path
        self.model_save_path = model_save_path
        self.df = None
        self.features_df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, data_path=None):
        """
        Load the raw transaction data from CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data file (overrides path provided at initialization)
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        if data_path:
            self.raw_data_path = data_path
            
        print(f"Loading data from: {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        
        # Drop any unnamed columns
        columns_to_drop = [col for col in self.df.columns if 'Unnamed' in col or col == 'index']
        self.df = self.df.drop(columns=columns_to_drop)
        
        print(f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self
    
    def clean_data(self):
        """
        Clean the loaded data by handling missing values, converting data types,
        and removing outliers.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Cleaning data...")
        
        # Convert SENDINGDATE to datetime
        self.df['SENDINGDATE'] = pd.to_datetime(self.df['SENDINGDATE'])
        
        # Convert SENDER_DATEOFBIRTH to datetime
        self.df['SENDER_DATEOFBIRTH'] = pd.to_datetime(self.df['SENDER_DATEOFBIRTH'])
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Fill categorical missing values with mode or special value
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna('Unknown')
        
        # Clean up email addresses
        if 'SENDER_EMAIL' in self.df.columns:
            self.df['SENDER_EMAIL'] = self.df['SENDER_EMAIL'].str.lower()
        
        if 'BENEFICIARY_EMAIL' in self.df.columns:
            self.df['BENEFICIARY_EMAIL'] = self.df['BENEFICIARY_EMAIL'].str.lower()
            
        print("Data cleaning completed")
        return self
    
    def engineer_features(self):
        """
        Create new features from the cleaned data to improve model performance.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Engineering features...")
        
        # Group by sender ID to get transaction metrics
        sender_metrics = self.df.groupby('SENDER_ID').agg({
            'MTN': 'count',  # Total transactions
            'BENEFICIARY_CLIENT_ID': pd.Series.nunique,  # Total beneficiaries
        }).rename(columns={
            'MTN': 'Total_Transactions',
            'BENEFICIARY_CLIENT_ID': 'Unique_Beneficiaries'
        })
        
        # Calculate transaction volume metrics per sender
        volume_metrics = self.df.groupby('SENDER_ID')['TOTALSALE'].agg([
            ('Avg_Transaction_Value', 'mean'),
            ('Max_Transaction_Value', 'max'),
            ('Min_Transaction_Value', 'min'),
            ('Std_Transaction_Value', 'std')
        ]).fillna(0)
        
        # Calculate paid out transactions ratio
        paid_out_counts = self.df[self.df['STATUS'] == 'Paid Out'].groupby('SENDER_ID').size()
        total_counts = self.df.groupby('SENDER_ID').size()
        paid_out_ratio = pd.DataFrame({
            'Paid_Out_Ratio': paid_out_counts / total_counts
        }).fillna(0)
        
        # Calculate the time patterns
        self.df['Day_of_Week'] = self.df['SENDINGDATE'].dt.dayofweek
        self.df['Hour_of_Day'] = self.df['SENDINGDATE'].dt.hour
        
        time_patterns = self.df.groupby('SENDER_ID').agg({
            'Day_of_Week': lambda x: x.value_counts().index[0] if len(x) > 0 else -1,
            'Hour_of_Day': lambda x: x.value_counts().index[0] if len(x) > 0 else -1
        }).rename(columns={
            'Day_of_Week': 'Most_Common_Day',
            'Hour_of_Day': 'Most_Common_Hour'
        })
        
        # Email domain matching with sender name
        if 'SENDER_EMAIL' in self.df.columns and 'SENDER_LEGALNAME' in self.df.columns:
            def check_email_name_match(row):
                if pd.isna(row['SENDER_EMAIL']) or row['SENDER_EMAIL'] == 'Unknown':
                    return 0
                
                if pd.isna(row['SENDER_LEGALNAME']) or row['SENDER_LEGALNAME'] == 'Unknown':
                    return 0
                
                # Extract domain from email
                try:
                    email_parts = row['SENDER_EMAIL'].split('@')
                    if len(email_parts) < 2:
                        return 0
                    
                    domain = email_parts[1].split('.')[0].lower()
                    
                    # Check if domain appears in legal name
                    sender_name = row['SENDER_LEGALNAME'].lower()
                    
                    if domain in sender_name or any(word in domain for word in sender_name.split()):
                        return 1
                    else:
                        return 0
                except:
                    return 0
            
            self.df['Email_Name_Match'] = self.df.apply(check_email_name_match, axis=1)
            email_metrics = self.df.groupby('SENDER_ID')['Email_Name_Match'].mean().reset_index()
            email_metrics = email_metrics.set_index('SENDER_ID')
        else:
            email_metrics = pd.DataFrame(index=sender_metrics.index)
            email_metrics['Email_Name_Match'] = 0
        
        # Calculate transaction spiking (top 5 daily transactions)
        def get_top_daily_trx(group):
            # Group by date and count transactions
            daily_counts = group.groupby(group['SENDINGDATE'].dt.date).size()
            # Get top 5 days
            if len(daily_counts) >= 5:
                top_5 = daily_counts.nlargest(5).tolist()
            else:
                # Pad with zeros if less than 5 days
                top_5 = daily_counts.nlargest(len(daily_counts)).tolist()
                top_5 = top_5 + [0] * (5 - len(top_5))
            return top_5
        
        transaction_spikes = self.df.groupby('SENDER_ID').apply(get_top_daily_trx)
        transaction_spikes = pd.DataFrame({
            'Top_5_Daily_Trx': transaction_spikes,
            'Avg_Top_5_Daily_Trx': transaction_spikes.apply(lambda x: sum(x) / 5 if len(x) > 0 else 0),
            'SD_Top_5_Trx': transaction_spikes.apply(lambda x: np.std(x) if len(x) > 0 else 0)
        })
        
        # Merge all features
        self.features_df = pd.concat([
            sender_metrics,
            volume_metrics,
            paid_out_ratio,
            time_patterns,
            email_metrics,
            transaction_spikes
        ], axis=1).fillna(0)
        
        # Add risk factors based on features
        self.features_df['Transaction_Risk_Score'] = (
            self.features_df['Total_Transactions'] / self.features_df['Unique_Beneficiaries'] *
            self.features_df['Std_Transaction_Value'] / (self.features_df['Avg_Transaction_Value'] + 1)
        ).fillna(0)
        
        # Prepare target column from the original dataset
        if 'Sender_Status' in self.df.columns:
            # Map Sender_Status to binary target (1 for Suspicious, 0 for Genuine)
            status_mapping = {
                'Genuine': 0,
                'Suspicious': 1
            }
            
            sender_status = self.df.groupby('SENDER_ID')['Sender_Status'].agg(
                lambda x: 1 if 'Suspicious' in x.values else 0 if 'Genuine' in x.values else np.nan
            )
            
            self.features_df['target'] = sender_status
            
            # Fill missing target values using rules-based approach
            missing_mask = self.features_df['target'].isna()
            
            # Rule: High risk score indicates suspicious activity
            risk_threshold = self.features_df['Transaction_Risk_Score'].quantile(0.95)
            self.features_df.loc[
                missing_mask & (self.features_df['Transaction_Risk_Score'] > risk_threshold),
                'target'
            ] = 1
            
            # Rule: Very typical behavior indicates genuine
            self.features_df.loc[
                missing_mask & (self.features_df['Transaction_Risk_Score'] < risk_threshold * 0.2),
                'target'
            ] = 0
            
            # Fill remaining with most common class (usually Genuine/0)
            most_common = self.features_df['target'].mode()[0]
            self.features_df['target'] = self.features_df['target'].fillna(most_common)
            
        print(f"Feature engineering completed. Created {self.features_df.shape[1]} features")
        return self
    
    def prepare_data_for_training(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets, and handle class imbalance.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Preparing data for training...")
        
        # Separate features and target
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"Data prepared. Training set: {self.X_train.shape[0]} samples, Test set: {self.X_test.shape[0]} samples")
        return self
    
    def train_model(self, model_type='random_forest'):
        """
        Train a model on the prepared data.
        
        Parameters:
        -----------
        model_type : str
            Type of model to train. Options: 'random_forest', 'logistic_regression', 'svm', 
            'knn', 'decision_tree', 'naive_bayes', 'ridge', 'voting'
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5, 
                min_samples_leaf=2, 
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        elif model_type == 'ridge':
            self.model = RidgeClassifier(random_state=42)
        elif model_type == 'voting':
            # Create a voting classifier with multiple models
            classifiers = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('dt', DecisionTreeClassifier(random_state=42))
            ]
            self.model = VotingClassifier(estimators=classifiers, voting='soft')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        print(f"{model_type} model training completed")
        return self
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_model(self, file_path=None):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        file_path : str
            Path where model will be saved (overrides path provided at initialization)
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        if file_path:
            self.model_save_path = file_path
            
        print(f"Saving model to: {self.model_save_path}")
        with open(self.model_save_path, 'wb') as file:
            pickle.dump(self.model, file)
            
        return self
    
    def load_model(self, file_path=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved model (overrides path provided at initialization)
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        if file_path:
            self.model_save_path = file_path
            
        print(f"Loading model from: {self.model_save_path}")
        with open(self.model_save_path, 'rb') as file:
            self.model = pickle.load(file)
            
        return self
    
    def predict_transaction(self, transaction_data):
        """
        Predict if a single transaction or a batch of transactions is suspicious.
        
        Parameters:
        -----------
        transaction_data : pd.DataFrame
            Transaction data to predict
            
        Returns:
        --------
        predictions : array
            Array of predictions (1 for suspicious, 0 for genuine)
        probabilities : array
            Array of probabilities of being suspicious
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        
        # Extract features from the transaction data
        # This is a simplified version - in practice, you would need to apply the same
        # feature engineering steps as during training
        features = self._extract_features_for_prediction(transaction_data)
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Get probabilities if the model supports it
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[:, 1]  # Probability of class 1 (suspicious)
        else:
            probabilities = predictions  # Just use the predictions if probabilities aren't available
        
        return predictions, probabilities
    
    def _extract_features_for_prediction(self, transaction_data):
        """
        Helper method to extract features from transaction data for prediction.
        
        Parameters:
        -----------
        transaction_data : pd.DataFrame
            Transaction data to extract features from
            
        Returns:
        --------
        features : array
            Array of features for prediction
        """
        # This is a placeholder implementation
        # In a real scenario, you would need to apply the same feature engineering
        # steps that were used during training
        
        # Example: Extract basic features
        features = []
        
        # Process each transaction
        for _, transaction in transaction_data.iterrows():
            # Example features (replace with actual feature engineering logic)
            transaction_features = [
                float(transaction.get('TOTALSALE', 0)),
                float(transaction.get('Day_of_Week', transaction.get('SENDINGDATE', datetime.now()).weekday())),
                float(transaction.get('Hour_of_Day', transaction.get('SENDINGDATE', datetime.now()).hour)),
                1.0 if transaction.get('STATUS', '') == 'Paid Out' else 0.0,
                # Add more features as needed
            ]
            features.append(transaction_features)
        
        # Convert to numpy array
        features = np.array(features)
        
        return features

    def run_pipeline(self, data_path=None, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Run the complete pipeline from data loading to model evaluation.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data file
        model_type : str
            Type of model to train
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        return (self
                .load_data(data_path)
                .clean_data()
                .engineer_features()
                .prepare_data_for_training(test_size, random_state)
                .train_model(model_type)
                .evaluate_model())


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline(
        raw_data_path="project_dataset_sw/project_dataset_sw.csv",
        model_save_path="random_forest_model.pkl"
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline(model_type='random_forest')
    
    # Save the trained model
    pipeline.save_model()
    
    print("Pipeline execution completed successfully.") 
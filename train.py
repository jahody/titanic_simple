"""
Simple Titanic ML Training Script
Loads data, trains a Random Forest model, and exports predictions
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import os


def load_data(train_path, test_path):
    """Load train and test data"""
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}, Test set: {test_df.shape}")

    # Store PassengerId for submission
    passenger_ids = test_df['PassengerId'].copy()

    # Use basic numeric features only
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    # Simple encoding: Sex to numeric
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    # Fill missing values with median
    for col in features:
        if col in train_df.columns:
            train_df[col].fillna(train_df[col].median(), inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(test_df[col].median(), inplace=True)

    X_train = train_df[features]
    y_train = train_df['Survived']
    X_test = test_df[features]

    print(f"Features: {features}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, y_train, X_test, passenger_ids


def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full dataset
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

    return model


def export_predictions(model, X_test, passenger_ids, output_path):
    """Generate and export predictions"""
    print(f"\nGenerating predictions...")
    predictions = model.predict(X_test)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    print(f"Exporting predictions to {output_path}...")
    submission.to_csv(output_path, index=False)
    print(f"Predictions exported successfully! ({len(predictions)} predictions)")

    # Show sample predictions
    print("\nSample predictions:")
    print(submission.head(10).to_string(index=False))


def main():
    """Main execution"""
    # Determine data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'data', 'train.csv')
    test_path = os.path.join(script_dir, 'data', 'test.csv')
    output_path = os.path.join(script_dir, 'predictions.csv')

    print("="*60)
    print("TITANIC SURVIVAL PREDICTION - RANDOM FOREST")
    print("="*60)

    # Step 1: Load data
    X_train, y_train, X_test, passenger_ids = load_data(train_path, test_path)

    # Step 2: Train model
    model = train_model(X_train, y_train)

    # Step 3: Export predictions
    export_predictions(model, X_test, passenger_ids, output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()

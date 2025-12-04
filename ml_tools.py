"""
ML Tools for Titanic Dataset
Implements all machine learning operations as tools for Claude Agent SDK
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, Any, List


class TitanicMLTools:
    """Container for all ML tools used by the agent"""

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, train_path: str, test_path: str) -> Dict[str, Any]:
        """Load training and test datasets"""
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)

            return {
                "success": True,
                "train_shape": self.train_df.shape,
                "test_shape": self.test_df.shape,
                "train_columns": list(self.train_df.columns),
                "test_columns": list(self.test_df.columns),
                "train_head": self.train_df.head(3).to_dict(),
                "missing_values_train": self.train_df.isnull().sum().to_dict(),
                "missing_values_test": self.test_df.isnull().sum().to_dict(),
                "target_distribution": self.train_df['Survived'].value_counts().to_dict() if 'Survived' in self.train_df.columns else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def explore_data(self, dataset: str = "train") -> Dict[str, Any]:
        """Explore dataset statistics"""
        df = self.train_df if dataset == "train" else self.test_df

        if df is None:
            return {"success": False, "error": "Data not loaded"}

        return {
            "success": True,
            "dataset": dataset,
            "shape": df.shape,
            "numeric_summary": df.describe().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_counts": df.isnull().sum().to_dict(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        }

    def preprocess_data(self, operations: List[str]) -> Dict[str, Any]:
        """Preprocess and engineer features"""
        if self.train_df is None or self.test_df is None:
            return {"success": False, "error": "Data not loaded"}

        try:
            # Work on copies
            train = self.train_df.copy()
            test = self.test_df.copy()

            operations_applied = []

            # Handle missing Age values
            if "fill_age" in operations:
                train['Age'].fillna(train['Age'].median(), inplace=True)
                test['Age'].fillna(test['Age'].median(), inplace=True)
                operations_applied.append("filled_age_with_median")

            # Handle missing Embarked
            if "fill_embarked" in operations:
                train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
                test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
                operations_applied.append("filled_embarked_with_mode")

            # Handle missing Fare
            if "fill_fare" in operations:
                test['Fare'].fillna(test['Fare'].median(), inplace=True)
                operations_applied.append("filled_fare_with_median")

            # Create FamilySize feature
            if "family_size" in operations:
                train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
                test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
                operations_applied.append("created_family_size")

            # Create IsAlone feature
            if "is_alone" in operations:
                train['IsAlone'] = (train['SibSp'] + train['Parch'] == 0).astype(int)
                test['IsAlone'] = (test['SibSp'] + test['Parch'] == 0).astype(int)
                operations_applied.append("created_is_alone")

            # Encode Sex
            if "encode_sex" in operations:
                train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
                test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
                operations_applied.append("encoded_sex")

            # Encode Embarked
            if "encode_embarked" in operations:
                embarked_dummies_train = pd.get_dummies(train['Embarked'], prefix='Embarked')
                embarked_dummies_test = pd.get_dummies(test['Embarked'], prefix='Embarked')
                train = pd.concat([train, embarked_dummies_train], axis=1)
                test = pd.concat([test, embarked_dummies_test], axis=1)
                operations_applied.append("encoded_embarked")

            # Extract Title from Name
            if "extract_title" in operations:
                train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
                test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

                # Consolidate rare titles
                title_mapping = {
                    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5
                }
                for df in [train, test]:
                    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                        'Jonkheer', 'Dona'], 'Rare')
                    df['Title'] = df['Title'].replace('Mlle', 'Miss')
                    df['Title'] = df['Title'].replace('Ms', 'Miss')
                    df['Title'] = df['Title'].replace('Mme', 'Mrs')
                    df['Title'] = df['Title'].map(title_mapping)
                    df['Title'].fillna(0, inplace=True)

                operations_applied.append("extracted_title")

            # Select features for training
            feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']

            # Add Title if created
            if 'Title' in train.columns:
                feature_cols.append('Title')

            # Add Embarked dummies if created
            if 'Embarked_C' in train.columns:
                feature_cols.extend(['Embarked_C', 'Embarked_Q', 'Embarked_S'])

            # Ensure all feature columns exist in both datasets
            available_features = [col for col in feature_cols if col in train.columns and col in test.columns]

            self.X_train = train[available_features]
            self.X_test = test[available_features]
            self.y_train = train['Survived']
            self.feature_names = available_features

            return {
                "success": True,
                "operations_applied": operations_applied,
                "feature_columns": available_features,
                "X_train_shape": self.X_train.shape,
                "X_test_shape": self.X_test.shape,
                "missing_after_preprocessing": {
                    "train": self.X_train.isnull().sum().to_dict(),
                    "test": self.X_test.isnull().sum().to_dict()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def train_model(self, model_type: str, hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a machine learning model"""
        if self.X_train is None or self.y_train is None:
            return {"success": False, "error": "Data not preprocessed"}

        try:
            if hyperparameters is None:
                hyperparameters = {}

            # Initialize model
            if model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 5),
                    random_state=42
                )
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(
                    max_iter=hyperparameters.get('max_iter', 1000),
                    random_state=42
                )
            elif model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 3),
                    learning_rate=hyperparameters.get('learning_rate', 0.1),
                    random_state=42
                )
            else:
                return {"success": False, "error": f"Unknown model type: {model_type}"}

            # Train model
            self.model.fit(self.X_train, self.y_train)

            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)

            # Training accuracy
            train_accuracy = self.model.score(self.X_train, self.y_train)

            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names,
                                             self.model.feature_importances_.tolist()))

            return {
                "success": True,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "train_accuracy": float(train_accuracy),
                "cv_scores": cv_scores.tolist(),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "feature_importance": feature_importance
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, output_path: str = None) -> Dict[str, Any]:
        """Generate predictions on test set"""
        if self.model is None:
            return {"success": False, "error": "Model not trained"}

        if self.X_test is None:
            return {"success": False, "error": "Test data not preprocessed"}

        try:
            # Generate predictions
            predictions = self.model.predict(self.X_test)

            # Create submission dataframe
            submission = pd.DataFrame({
                'PassengerId': self.test_df['PassengerId'],
                'Survived': predictions
            })

            # Save if path provided
            if output_path:
                submission.to_csv(output_path, index=False)

            return {
                "success": True,
                "predictions_count": len(predictions),
                "predictions_sample": predictions[:10].tolist(),
                "survival_rate": float(predictions.mean()),
                "output_path": output_path,
                "submission_preview": submission.head(10).to_dict()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Tool definitions for Claude Agent SDK
TOOL_DEFINITIONS = [
    {
        "name": "load_data",
        "description": "Load the Titanic training and test CSV datasets. Returns dataset shapes, columns, missing value counts, and sample data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "train_path": {
                    "type": "string",
                    "description": "Path to the training CSV file (train.csv)"
                },
                "test_path": {
                    "type": "string",
                    "description": "Path to the test CSV file (test.csv)"
                }
            },
            "required": ["train_path", "test_path"]
        }
    },
    {
        "name": "explore_data",
        "description": "Explore dataset statistics including summary statistics, data types, missing values, and column types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "enum": ["train", "test"],
                    "description": "Which dataset to explore: 'train' or 'test'"
                }
            },
            "required": ["dataset"]
        }
    },
    {
        "name": "preprocess_data",
        "description": "Preprocess data and engineer features. Operations include: fill_age, fill_embarked, fill_fare, family_size, is_alone, encode_sex, encode_embarked, extract_title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["fill_age", "fill_embarked", "fill_fare", "family_size",
                                "is_alone", "encode_sex", "encode_embarked", "extract_title"]
                    },
                    "description": "List of preprocessing operations to apply"
                }
            },
            "required": ["operations"]
        }
    },
    {
        "name": "train_model",
        "description": "Train a machine learning model for survival prediction. Supports random_forest and logistic_regression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["random_forest", "logistic_regression"],
                    "description": "Type of model to train"
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "Optional hyperparameters for the model",
                    "properties": {
                        "n_estimators": {"type": "integer"},
                        "max_depth": {"type": "integer"},
                        "max_iter": {"type": "integer"}
                    }
                }
            },
            "required": ["model_type"]
        }
    },
    {
        "name": "predict",
        "description": "Generate survival predictions on the test dataset and optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save predictions CSV file"
                }
            }
        }
    }
]


def process_tool_call(tool_name: str, tool_input: Dict[str, Any], ml_tools: TitanicMLTools) -> Dict[str, Any]:
    """Route tool calls to appropriate methods"""
    if tool_name == "load_data":
        return ml_tools.load_data(tool_input['train_path'], tool_input['test_path'])
    elif tool_name == "explore_data":
        return ml_tools.explore_data(tool_input['dataset'])
    elif tool_name == "preprocess_data":
        return ml_tools.preprocess_data(tool_input['operations'])
    elif tool_name == "train_model":
        return ml_tools.train_model(
            tool_input['model_type'],
            tool_input.get('hyperparameters')
        )
    elif tool_name == "predict":
        return ml_tools.predict(tool_input.get('output_path'))
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

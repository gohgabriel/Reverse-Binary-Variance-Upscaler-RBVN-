import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DataPreprocessor:
    def __init__(self, data, target_column, categorical_columns=None, train_size=0.9, positive_label=None):
        self.data = data
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.train_size = train_size
        self.positive_label = positive_label
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def is_target_binary(self):
        unique_values = self.data[self.target_column].unique()
        return (len(unique_values) == 2) and all(value in [0, 1] for value in unique_values)

    def convert_target_to_binary(self):
        unique_values = self.data[self.target_column].unique()
        if len(unique_values) != 2:
            raise ValueError(f"Target column '{self.target_column}' must have exactly two unique values.")
        if self.positive_label is None:
            self.positive_label = unique_values[1]
        elif self.positive_label not in unique_values:
            raise ValueError(f"Specified positive label '{self.positive_label}' is not found in the target column.")
        self.data[self.target_column] = np.where(self.data[self.target_column] == self.positive_label, 1, 0)

    def detect_categorical_columns(self):
        print("Detecting categorical columns...")

        if self.categorical_columns is None:
            self.categorical_columns = []
            for column in self.data.columns:
                print(f"Column: {column}, Data Type: {self.data[column].dtype}")
                if self.data[column].dtype == 'object' and column != self.target_column:
                    self.categorical_columns.append(column)

        print("Detected categorical columns:", self.categorical_columns)

    def encode_data(self):
        if self.categorical_columns:
            data_encoded = pd.get_dummies(self.data, columns=self.categorical_columns)
        else:
            data_encoded = self.data.copy()

        data_encoded = data_encoded.astype(int)
        self.X = data_encoded.drop(self.target_column, axis=1)
        self.y = data_encoded[self.target_column]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=self.train_size, shuffle=True
        )

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, device, custom_model=None):
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device).reshape(-1, 1)
        self.X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device).reshape(-1, 1)
        self.device = device
        self.model = None
        self.best_weights = None
        self.best_f1 = 0.0
        self.history = []
        self.custom_model = custom_model

    def build_model(self):
        if self.custom_model:
            print("Custom model specified. Using custom model...")
            input_size = self.X_train.shape[1]
            self.model = CustomModel(input_size, self.custom_model)
        else:

            print("No custom model specified. Using default model...")

            x_cols = self.X_train.shape[1]
            self.model = nn.Sequential(
                nn.Linear(x_cols, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(8, 1)
        )
        self.model.to(self.device)

    def train_model(self, n_epochs=100, batch_size=16, patience=10, delta=0.001):
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        batch_start = torch.arange(0, len(self.X_train), batch_size)

        best_f1 = 0.0
        best_weights = None
        early_stopping_counter = 0
        f1_history = []

        for epoch in range(n_epochs):
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    X_batch = self.X_train[start:start+batch_size]
                    y_batch = self.y_train[start:start+batch_size]

                    y_pred_proba = self.model(X_batch)
                    loss = loss_fn(y_pred_proba, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    bar.set_postfix(loss=float(loss))

            with torch.no_grad():
                y_pred_proba = self.model(self.X_test)
                y_pred = (y_pred_proba > 0).float()
                f1 = f1_score(self.y_test.cpu(), y_pred.cpu())
                f1_history.append(f1)

            if f1 > best_f1 + delta:
                best_f1 = f1
                best_weights = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch} with F1 score: {best_f1:.4f}")
                    break

        # Restore model with best weights and save it
        self.model.load_state_dict(best_weights)
        torch.save(self.model, 'my_trained_model.pt')

        print(f"Best F1: {best_f1:.4f}")
        plt.plot(f1_history)
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.title('Training F1 History')
        plt.show()

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            y_pred_proba = self.model(self.X_test)
            y_pred = (y_pred_proba > 0).float()
            accuracy = accuracy_score(self.y_test.cpu(), y_pred.cpu())
            precision = precision_score(self.y_test.cpu(), y_pred.cpu())
            recall = recall_score(self.y_test.cpu(), y_pred.cpu())
            f1 = f1_score(self.y_test.cpu(), y_pred.cpu())
        return accuracy, precision, recall, f1

class CustomModel(nn.Module):
    def __init__(self, input_size, model_architecture):
        super().__init__()
        layers = list(model_architecture)
        layers[0] = layers[0].__class__(input_size, layers[0].out_features)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_probabilities(self, new_data_features):
        new_data_tensor = torch.tensor(new_data_features.values, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(new_data_tensor)
            probabilities = torch.sigmoid(logits)
        return probabilities.cpu().numpy()

def convert_binary_to_probability(data, target_column, categorical_columns=None, train_size=0.9, n_epochs=100, batch_size=16, threshold=0.5, positive_label=None, custom_model=None):
    try:
        print("\033[1mStep 1: Checking CUDA availability\033[0m")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available! Using GPU")
        else:
            device = torch.device('cpu')
            print("CUDA not available. Using CPU")

        print("\n\033[1mStep 2: Data preprocessing\033[0m")
        data_copy = data.copy()  # Create a copy of the input datafile
        preprocessor = DataPreprocessor(data_copy, target_column, categorical_columns, train_size, positive_label)

        # Check if the target column is already binary
        if not preprocessor.is_target_binary():
            preprocessor.convert_target_to_binary()
        else:
            print(f"Target column '{target_column}' is already binary. Skipping conversion.")

        preprocessor.detect_categorical_columns()
        preprocessor.encode_data()
        preprocessor.split_data()

        print("\n\033[1mStep 3: Model training\033[0m")
        trainer = ModelTrainer(preprocessor.X_train, preprocessor.y_train, preprocessor.X_test, preprocessor.y_test,
                               device, custom_model)
        trainer.build_model()
        trainer.train_model(n_epochs=n_epochs, batch_size=batch_size)

        print("\n\033[1mStep 4: Model evaluation\033[0m")
        accuracy, precision, recall, f1 = trainer.evaluate_model()
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

        print("\n\033[1mStep 5: Prediction\033[0m")

        print("Upscaling binary categories to probabilities...")

        print("Disclaimer: If F1 score is low, data may not contain sufficient information to create accurate binary-to-probability upscaling.")

        new_data_features = preprocessor.data.drop(target_column, axis=1)

        if preprocessor.categorical_columns:
            new_data_features = pd.get_dummies(new_data_features, columns=preprocessor.categorical_columns, dtype=int)
            new_data_features = new_data_features.reindex(columns=preprocessor.X.columns, fill_value=0)

        predictor = Predictor(trainer.model, device)
        probabilities_numpy = predictor.predict_probabilities(new_data_features)
        predicted_labels = np.where(probabilities_numpy >= threshold, 1, 0)

        print("\n\033[1mStep 6: Creating results DataFrame\033[0m")
        print("Labeling predictions...")

        results_df = pd.DataFrame({
            f'{target_column}_rbprob': probabilities_numpy.flatten(),
            f'{target_column}_rblab': predicted_labels.flatten()
        })

        print("\n\033[1mStep 7: Concatenating results with original data\033[0m")
        print("Creating output data frame...")
        output_df = pd.concat([data, results_df], axis=1)  # Concatenate original data with predictions

        return output_df

    except ValueError as ve:
        print("ValueError occurred:")
        print(str(ve))

    except Exception as e:
        print("An unexpected error occurred:")
        print(str(e))
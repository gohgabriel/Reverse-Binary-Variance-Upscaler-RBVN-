# Reverse Binary Variance Upscaling (RBVN)

This Python script performs Reverse Binary Variance Upscaling (RBVN), a technique to convert binary target variables into probability scores based on the input features. It uses a neural network model to learn the mapping from input features to the binary target, and then leverages the model's output logits to generate probability scores. These scores can then be used for predictive analysis for variables that the model was not trained on (to prevent leakage).

The code is flexible and allows the user to define their own model if the inbuilt model is not sufficient or has poor performance.

RBVN's major use-case is for exploratory and preliminary data analysis. It is particularly useful when the central aim is to create a composite variable from different measures of a single construct used in different studies (e.g. binary versus interval measurements of a single construct). RBVN allows for binary constructs to be "upscaled" to interval variables, which can then be subject to POMP scaling for meta-analysis.

## Features

- Preprocess data by detecting and encoding categorical features
- Train a neural network model on the binary target variable
- Evaluate the trained model's performance
- Predict probability scores for the binary target based on input features
- Combine the original data with predicted probabilities and binary labels

## Usage

1. Import the necessary libraries and define any custom model architectures.
2. Call the `convert_binary_to_probability` function with your data, target column name, and any additional parameters.
3. The function returns a DataFrame containing the original data along with the predicted probabilities and binary labels.

        python
        import pandas as pd
        from reverse_binary import convert_binary_to_probability
        
        # Load your data
        data = pd.read_csv('your_data.csv')
        
        # Define the target column name
        target_column = 'target_variable'
        
        # Call the conversion function
        output_df = convert_binary_to_probability(data, target_column)

## Parameters

      data (pandas.DataFrame): Input data containing features and the binary target variable.
      target_column (str): Name of the column containing the binary target variable.
      categorical_columns (list, optional): List of column names to be treated as categorical features. (default: None, automatically handled by the script)
      train_size (float, optional): Fraction of the data to be used for training (default: 0.9).
      n_epochs (int, optional): Number of epochs for model training (default: 100).
      batch_size (int, optional): Batch size for model training (default: 16).
      threshold (float, optional): Threshold for converting probability scores to binary labels (default: 0.5).
      positive_label (int or str, optional): Value representing the positive class in the binary target (default: None, inferred from data).
      custom_model (nn.Module, optional): Custom PyTorch model architecture (default: None, uses a default architecture).

## Output
The function returns a pandas DataFrame containing the following columns:

      Original data columns
      {target_column}_rbprob: Predicted probability scores for the target variable
      {target_column}_rblab: Predicted binary labels based on the threshold

## Requirements

      Python 3.6 or later
      pandas
      numpy
      torch
      scikit-learn

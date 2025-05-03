import pandas as pd
import numpy as np

def calculate_mae(csv_file, actual_col, predicted_col):
    df = pd.read_csv(csv_file)
    actual = df[actual_col].values
    predicted = df[predicted_col].values
    mae = np.mean(np.abs(actual - predicted))
    return mae

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate MAE from CSV file with actual and predicted values')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--actual_col', default='actual_length', help='Name of the column with actual values')
    parser.add_argument('--predicted_col', default='predicted_label', help='Name of the column with predicted values')

    args = parser.parse_args()

    mae = calculate_mae(args.csv_file, args.actual_col, args.predicted_col)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate MSE from CSV file with actual and predicted values')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--actual_col', default='actual_length', help='Name of the column with actual values')
    parser.add_argument('--predicted_col', default='predicted_label', help='Name of the column with predicted values')
    
    args = parser.parse_args()
    
    # Calculate MSE
    mse = calculate_mae(args.csv_file, args.actual_col, args.predicted_col)
    
    # Print results
    print(f"Mean Squared Error (MSE): {mse:.4f}")


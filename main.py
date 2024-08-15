import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from multi_rnn import MultiRNN
import argparse


def main(args):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(args.file_path, parse_dates=True, index_col="date")
    # Select data from a specific date onwards
    df = df.loc[args.start_date :]

    # Print the column names of the DataFrame
    print(f"Columns provided: {df.columns}")
    # Print the number of rows in the DataFrame
    print(f"Number of rows: {len(df)}")

    # Define a variable for testing purposes
    test = 1
    # Calculate the number of rows needed for the test data based on a time interval of 10 minutes
    rows = int(24 * 60 / 3)
    # Calculate the index of the test data based on the test variable and the number of rows
    test_index = test * rows
    # Create the training data by selecting all rows except the test data
    train = df.iloc[:-test_index]
    # Create the test data by selecting the last rows based on the test_index
    test = df.iloc[-test_index:]

    # Create an instance of the MultiRNN class with specified parameters
    my_rnn = MultiRNN(
        train=train,
        test=test,
        length=10,
        LSTM_units=64,
        activation="tanh",
        batch_size=1,
        epochs=2,
    )

    predictions = my_rnn.predict(df)

    # Define the column name to plot
    column_name = "Appliances"
    # Define the width and height of the plot
    width = 10
    height = 8
    # Define the name to save the plot as
    save_name = "predict_vs_test_plot"
    save_name_2 = "loss_val_loss_plot"

    # Call the plot_predict_against_test_dataset_per_column function and pass in the necessary arguments
    my_rnn.plot_predict_against_test_dataset_per_column(
        column_name, width, height, save_name
    )

    # Call the plot_loss_val_loss_per_column function and pass in the necessary arguments
    my_rnn.plot_loss_val_loss_per_column(column_name, width, height, save_name_2)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Description of your script")

    # Add command-line arguments
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    parser.add_argument("start_date", type=str, help="Start date for data selection")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)

    # python main.py updated_energydata.csv "2016-05-24"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt


class MultiRNN:
    """
    A class for creating and training multi-feature recurrent neural network models.

    Parameters:
    train (pd.DataFrame): The training data as a pandas DataFrame.
    test (pd.DataFrame): The test data as a pandas DataFrame.
    length (int, optional): The length of input sequences for the RNN models. Defaults to 1.
    LSTM_units (int, optional): The number of LSTM units in each layer of the RNN models. Defaults to 100.
    activation (str, optional): The activation function to use in the LSTM layers. Allowed values are "tanh", "sigmoid", "softmax", and "relu". Defaults to "tanh".
    optimizer (str, optional): The optimizer algorithm to use for training the RNN models. Allowed values are "adam", "rmsprop", and "sgd". Defaults to "adam".
    batch_size (int, optional): The batch size for training the RNN models. Defaults to 1.
    epochs (int, optional): The number of epochs to train the RNN models. Defaults to 25.

    Raises:
    ValueError: If the train or test data contains missing or categorical values.
    ValueError: If length is not a positive integer.
    ValueError: If LSTM_units is not a positive integer.
    ValueError: If activation is not one of the allowed values.
    ValueError: If optimizer is not one of the allowed values.
    ValueError: If batch_size is not a positive integer.
    ValueError: If epochs is not a positive integer.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        length: int = 1,
        LSTM_units: int = 100,
        activation: str = "tanh",
        optimizer: str = "adam",
        batch_size: int = 1,
        epochs: int = 25,
    ) -> None:
        # Check if train data is ready for processing
        if self.ready_for_processing(train):
            self.train = train
            print("train OK")
        else:
            raise ValueError("Train data contains missing and/or categorical values.")

        # Check if test data is ready for processing
        if self.ready_for_processing(test):
            self.test = test
            print("test OK")
        else:
            raise ValueError("Test data contains missing and/or categorical values.")

        # Validate and assign the input parameters
        if length > 0 and isinstance(length, int):
            self.length = length
            print("length OK")
        else:
            raise ValueError("length has to be a positive integer number.")

        if LSTM_units > 0 and isinstance(LSTM_units, int):
            self.LSTM_units = LSTM_units
            print("LSTM OK")
        else:
            raise ValueError("LSTM_units has to be a positive integer number.")

        if activation in ("tanh", "sigmoid", "softmax", "relu"):
            self.activation = activation
            print("activation OK")
        else:
            raise ValueError(
                'activation has to be "tanh", "sigmoid", "softmax" or "relu".'
            )

        if optimizer in ("adam", "rmsprop", "sgd"):
            self.optimizer = optimizer
            print("optimizer OK")
        else:
            raise ValueError('optimizer has to be "adam","rmsprop" or "sgd".')

        if batch_size > 0 and isinstance(batch_size, int):
            self.batch_size = batch_size
            print("batch_size OK")
        else:
            raise ValueError("batch_size has to be a positive integer number.")

        if batch_size > 0 and isinstance(batch_size, int):
            self.epochs = epochs
            print("epochs OK")
        else:
            raise ValueError("epochs has to be a positive integer number.")

        self.model_dict = self.generate_model_list_per_column_in_data_set(
            train, test, length, LSTM_units, activation, optimizer, batch_size, epochs
        )

    def ready_for_processing(self, data_set: pd.DataFrame):
        """
        Checks if the given dataset is ready for processing.

        Parameters:
        - data_set (pd.DataFrame): The dataset to be checked.

        Returns:
        - bool: True if the dataset is ready for processing, False otherwise.

        The function checks if the dataset has no missing values and contains only numeric data types.
        If both conditions are met, it returns True. Otherwise, it returns False.
        """
        # Check for missing values
        if data_set.isnull().any().any():
            return False

        # Check for object data type
        if data_set.select_dtypes(include=["object"]).columns.tolist():
            return False

        return True

    def generate_data_set_per_column_with_original_index(
        self, data_set: pd.DataFrame, save: bool = False
    ) -> dict:
        """
        Generates a dictionary of pandas DataFrames, where each DataFrame contains a single column from the input dataset.
        The index of each DataFrame is set to match the original dataset's index.

        Args:
            data_set (pd.DataFrame): The input dataset from which to extract columns.
            save (bool, optional): Flag indicating whether to save each DataFrame as a CSV file. Defaults to False.

        Returns:
            dict: A dictionary of pandas DataFrames, where each DataFrame contains a single column from the input dataset.
        """
        df_dict = {}  # Dictionary to store the generated data frames

        for col in data_set.columns:
            new_data = data_set[
                [col]
            ].copy()  # Create a new data frame with a single column
            new_data.index = (
                data_set.index
            )  # Set the index to match the original data set's index

            df_dict[col] = new_data  # Add the new data frame to the dictionary

            if save:
                csv_name = "RNN_" + col + "_column.csv"  # Generate a CSV file name
                new_data.to_csv(
                    csv_name, index_label="timestamp"
                )  # Save the data frame as a CSV file
                print(f"[{col}]: data frame created successfully and saved.")
            else:
                print(f"[{col}]: data frame created successfully.")

        self.dict_of_data_frames = (
            df_dict  # Store the dictionary of data frames in the class variable
        )

        return df_dict  # Return the dictionary of generated data frames

    def build_model_per_column(
        self, train, test, length, LSTM_units, activation, optimizer, batch_size, epochs
    ):
        """
        Builds an LSTM model to predict values for each column in the dataset.

        Args:
            train (DataFrame): The training data.
            test (DataFrame): The test data.
            length (int): The length of the input sequences for the LSTM model.
            LSTM_units (int): The number of LSTM units in the model.
            activation (str): The activation function for the LSTM layer.
            batch_size (int): The batch size for training and evaluation.
            epochs (int): The number of epochs to train the model.

        Returns:
            tuple: A tuple containing the trained model, the losses dataframe, and the model name.
        """
        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)

        # Create generators for training and testing data
        train_generator = TimeseriesGenerator(
            scaled_train, scaled_train, length=length, batch_size=batch_size
        )
        test_generator = TimeseriesGenerator(
            scaled_test, scaled_test, length=length, batch_size=batch_size
        )

        n_features = len(train.columns)

        # Build the LSTM model
        model = Sequential()
        model.add(
            LSTM(
                units=LSTM_units,
                activation=activation,
                input_shape=(length, scaled_train.shape[1]),
            )
        )
        model.add(Dense(n_features))
        model.compile(optimizer=optimizer, loss="mse")
        # Define early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor="val_loss", patience=2)

        # Train the model
        results = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[early_stop],
            verbose=1,
        )

        test_predictions = []

        # Generate predictions for the test data
        eval_first_batch = scaled_train[-length:]
        current_batch = eval_first_batch.reshape((1, length, n_features))

        for _ in range(len(test)):
            # Predict the next value based on the current batch
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)

            # Update the current batch by shifting one step forward and appending the predicted value
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform the scaled predictions
        predictions = scaler.inverse_transform(test_predictions)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv("predictions.csv")

        # Save the losses during training
        losses = pd.DataFrame(results.history)
        losses.to_csv("losses.csv")

        # Save the trained model
        model_name = "self.model.h5"
        model.save(model_name)

        return model, losses, model_name

    def generate_model_list_per_column_in_data_set(
        self, train, test, length, LSTM_units, activation, optimizer, batch_size, epochs
    ):
        """
        Generates a dictionary of models per column in the dataset.

        Args:
            train (DataFrame): The training dataset.
            test (DataFrame): The testing dataset.
            length (int): Length of the data.
            LSTM_units (int): Number of LSTM units in the model.
            activation (str): Activation function for the model.
            batch_size (int): Batch size for training the model.
            epochs (int): Number of epochs for training the model.

        Returns:
            dict: A dictionary containing the models, losses, and model names for each column in the dataset.
        """
        self.model_dict = {}

        # Generate data set per column for train and test
        train_data_columns = self.generate_data_set_per_column_with_original_index(
            train
        )
        test_data_columns = self.generate_data_set_per_column_with_original_index(test)

        for column, train_data in train_data_columns.items():
            print(f"Current column in training: {column}")
            test_data = test_data_columns[column]

            # Build the model per column
            model, losses, model_name = self.build_model_per_column(
                train_data,
                test_data,
                length,
                LSTM_units,
                activation,
                optimizer,
                batch_size,
                epochs,
            )

            # Add the model, losses, and model_name to the dictionary
            self.model_dict[column] = {
                "model": model,
                "losses": losses,
                "model_names": model_name,
            }

        return self.model_dict

    def predict(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts values for each column in the input DataFrame using the corresponding trained models.

        Args:
            data_frame (pd.DataFrame): The input DataFrame containing the data to be predicted.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted values for each column in the input DataFrame.
        """
        # Initialize an empty numpy array to store the predicted values for each column
        predictions = np.empty((data_frame.shape[0], 0))

        # Check if the input data_frame is a single-column DataFrame
        if data_frame.shape[1] == 1:
            # Perform predictions for the single column using the corresponding model
            column = data_frame.columns[0]
            column_prediction = self.model_dict[column]["model"].predict(
                data_frame.values.reshape(-1, 1)
            )
            predictions = np.hstack((predictions, column_prediction))
        else:
            # Iterate over the columns
            for column in data_frame.columns:
                # Perform predictions for the current column using the corresponding model
                column_prediction = self.model_dict[column]["model"].predict(
                    data_frame[column].values.reshape(-1, 1)
                )
                predictions = np.hstack((predictions, column_prediction))

        # Convert the numpy array of predictions into a DataFrame
        pred_df = pd.DataFrame(
            predictions, index=data_frame.index, columns=data_frame.columns
        )

        # Return the DataFrame with predicted values
        return pred_df

    def plot_predict_against_test_dataset_per_column(
        self,
        column: str,
        figure_width: int,
        figure_height: int,
        save_plot_name: str = "plot_test_vs_predict",
    ):
        """
        Plots the test values and predicted values for a specified column against the test dataset.

        Args:
            column (str): The column name for which to plot the test and predicted values.
            figure_width (int): The width of the plot figure in inches.
            figure_height (int): The height of the plot figure in inches.
            save_plot_name (str, optional): The base name for saving the plot image file. Defaults to "plot_test_vs_predict_".

        Returns:
            None: The plot is displayed and saved as an image file.
        """
        # Extract the test data for the specified column
        test_data = self.test[column]

        # Predict the values for the test dataset
        predicted_data = self.predict(self.test)[column]

        # Create a new figure with the specified dimensions
        plt.figure(figsize=(figure_width, figure_height))
        # Plot the test values as a blue line
        plt.plot(test_data, color="blue", label="Test")
        # Plot the predicted values as a red line
        plt.plot(predicted_data, color="red", label="Predicted")
        # Set title for the plot
        plt.title(f"Test vs Predicted ({column})")
        # Display a legend
        plt.legend()
        # Save the plot with the specified name
        plt.savefig(save_plot_name + column + ".png")
        # Show the plot
        plt.show()

    def plot_loss_val_loss_per_column(
        self,
        column: str,
        figure_width: int,
        figure_height: int,
        save_plot_name: str = "Plot_Loss_val_loss_",
    ):
        """
        Plots the loss and val_loss values for a specified column from the model_dict.

        Args:
            column (str): The column name for which the losses should be plotted.
            figure_width (int): The width of the plot figure in inches.
            figure_height (int): The height of the plot figure in inches.
            save_plot_name (str, optional): The base name for the saved plot file. Defaults to "Plot_Loss_val_loss_".

        Returns:
            None: Displays the plot and saves it as a PNG file.
        """
        # Get the losses DataFrame for the specified column from the model_dict
        losses_df = self.model_dict[column]["losses"]

        # Create a new figure with the specified dimensions
        plt.figure(figsize=(figure_width, figure_height))
        # Plot the loss values as a blue line
        plt.plot(losses_df["loss"], color="blue", label="loss")
        # Plot the val_loss values as a red line
        plt.plot(losses_df["val_loss"], color="red", label="val_Loss")
        # Set title for the plot
        plt.title(f"loss vs val_Loss ({column})")
        # Display a legend
        plt.legend()
        # Save the plot with the specified name
        plt.savefig(save_plot_name + column + ".png")
        # Show the plot
        plt.show()

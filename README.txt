
███╗░░░███╗██╗░░░██╗██╗░░░░░████████╗██╗  ██████╗░███╗░░██╗███╗░░██╗
████╗░████║██║░░░██║██║░░░░░╚══██╔══╝██║  ██╔══██╗████╗░██║████╗░██║
██╔████╔██║██║░░░██║██║░░░░░░░░██║░░░██║  ██████╔╝██╔██╗██║██╔██╗██║
██║╚██╔╝██║██║░░░██║██║░░░░░░░░██║░░░██║  ██╔══██╗██║╚████║██║╚████║
██║░╚═╝░██║╚██████╔╝███████╗░░░██║░░░██║  ██║░░██║██║░╚███║██║░╚███║
╚═╝░░░░░╚═╝░╚═════╝░╚══════╝░░░╚═╝░░░╚═╝  ╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚══╝


MultiRNN
    MultiRNN is a Python class that implements a recurrent neural network (RNN) model for multi-feature prediction.
    It provides a convenient way to build and train LSTM models, handling data preprocessing, model creation, training, and prediction.
    It was developed as part of a school project and is designed to leverage the Keras library for building and training RNN models.
    The program takes training and test data as input, along with customizable parameters such as sequence length, number of LSTM units,activation function, optimizer algorithm, batch size, and number of epochs.
    It builds separate RNN models for each column in the dataset and trains them on the provided data. The trained models can be used for predicting values on new input data.

Features
   • Creates and trains multi-feature recurrent neural network models.
   • Handles missing values and ensures input data is numeric.
   • Supports customization of model parameters such as sequence length, number of LSTM units, activation function, optimizer, batch size, and epochs.
   • Provides methods for generating predictions and visualizing results.

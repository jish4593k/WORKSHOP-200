

import os
import pandas as pd
import csv
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Check if data.csv exists, if not, create it
def check_if_exist():
    if not os.path.isfile('data.csv'):
        starting_data = []
        df = pd.DataFrame(starting_data, columns=['folder', 'file', 'path', 'data'])
        df.to_csv("data.csv", index=False)

check_if_exist()

# Function to create a simple linear regression model
def train_regression_model(X, y):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    return model

# GUI class for interactive functionalities
class DataLoggerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.path = "root/"
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Data Logger")
        self.command_input = tk.Entry(self.root, width=50)
        self.command_input.grid(row=0, column=0, padx=10, pady=10)
        self.command_input.insert(0, "Hello. you are at: root/")

        self.execute_button = tk.Button(self.root, text="Execute", command=self.execute_command)
        self.execute_button.grid(row=0, column=1, padx=10, pady=10)

    def execute_command(self):
        command_input = self.command_input.get()
        self.process_command(command_input)

    def process_command(self, command_input):
        # Your existing logic for processing commands

        # For example, let's add regression functionality to predict data
        if command_input.startswith('predict'):
            model = self.train_regression_example()
            self.predict_and_display()

    def train_regression_example(self):
        # Dummy regression data, replace with your actual dataset
        X_regression = np.random.rand(100, 1)
        y_regression = 3 * X_regression.flatten() + 2 + 0.1 * np.random.randn(100)

        # Train a regression model
        model = train_regression_model(X_regression, y_regression)
        return model

    def predict_and_display(self):
        # Example: Predict using the trained regression model
        features = np.array([[0.5]])
        prediction = model.predict(features)
        print("Regression Prediction:", prediction[0, 0])


class DataLogger:
    def __init__(self):
        self.folders = []
        self.files = []

  
    def data_mining_example(self):
        data = pd.read_csv('data.csv')
        sns.pairplot(data)
        plt.show()


data_logger = DataLogger()
gui = DataLoggerGUI()

gui.root.mainloop()

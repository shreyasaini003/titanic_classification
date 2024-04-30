# Titanic Survival Prediction

This repository contains a simple neural network model implemented in Python using TensorFlow to predict survival on the Titanic based on passenger data. The model is trained on the famous Titanic dataset which contains information about passengers such as age, sex, ticket fare, etc.

## Dataset

The dataset used in this project consists of two CSV files:
- `train_dataset.csv`: Training dataset containing features and survival labels.
- `test_dataset.csv`: Test dataset containing features without survival labels.

## Installation

To run the code in this repository, you need to have Python installed along with the following libraries:
- pandas
- numpy
- tensorflow

You can install these dependencies using pip:
pip install pandas numpy tensorflow


## Usage

1. Clone this repository:
git clone https://github.com/shreyasaini003/titanic_classification


2. Navigate to the cloned directory:
cd titanic_classification


## Model Architecture

The neural network model architecture used in this project is as follows:
- Input layer with 6 neurons corresponding to 6 features
- One hidden layer with 32 neurons and ReLU activation function
- One hidden layer with 16 neurons and ReLU activation function
- Output layer with 1 neuron and sigmoid activation function

## Results

After training the model, it is evaluated on the test set to measure its accuracy. Additionally, predictions can be made on new data using the trained model.

## Example

An example of making predictions on new data is provided in the script. You can modify the `new_data` array to input different feature values and see the model's survival prediction.




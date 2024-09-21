# AI_Hackathon
Stock Price Forecasting
# Stock Price Prediction with LSTM
## Overview

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. The model is designed to predict the closing prices of Microsoft (MSFT) stock based on historical price data. The project involves data preprocessing, feature scaling, model building, training, and evaluation.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- **Python**: The primary programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib**: For data visualization.
- **scikit-learn**: For model evaluation and preprocessing.
- **TensorFlow**: For building and training the LSTM neural network.

## Data Description

The dataset used in this project is the historical stock prices of Microsoft Corporation (MSFT), obtained in CSV format. The relevant features include:

- **Date**: The date of the stock price entry.
- **Close**: The closing price of the stock on that date.

## Installation

To run this project, ensure you have Python 3.x installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Abhishek0-7/AI_Hackathon.git
   cd stock-price-prediction
   ```

2. **Download the dataset:**

   Place the `MSFT.csv` file in the root directory of the project.

3. **Run the code:**

   Execute the Python script to train the model and make predictions:

   ```bash
   python stock_price_prediction.py
   ```

## Model Architecture

The LSTM model architecture consists of:

- An input layer accepting sequences of stock prices.
- Two LSTM layers with 256 and 128 units, respectively, for capturing temporal dependencies.
- Batch normalization and dropout layers for regularization.
- A dense layer with ReLU activation followed by a final output layer producing the predicted closing price.

## Results

The model's performance is evaluated using root mean square error (RMSE) and mean absolute percentage error (MAPE). The results will include visualizations of the predicted vs. actual stock prices for training, validation, and test datasets.

## Evaluation Metrics

- **RMSE (Root Mean Square Error)**: Indicates the average deviation of the predicted values from the actual values.
- **MAPE (Mean Absolute Percentage Error)**: Provides accuracy in percentage, indicating how close the predictions are to the actual values.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

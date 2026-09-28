# Car Price Prediction with Machine Learning

## CodeAlpha Data Science Internship – Task 3

### Objective

The objective of this project is to predict the selling price of used cars using machine learning.

### Dataset

The dataset contains information about used cars, including features such as:

* Car Name
* Year
* Selling Price
* Present Price
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission
* Previous Owners

### Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

### Methodology

The project follows these steps:

1. Load the car dataset.
2. Explore the data and check missing values.
3. Separate input features and the target selling price.
4. Identify numerical and categorical features.
5. Convert categorical data using One-Hot Encoding.
6. Split the dataset into training and testing data.
7. Train a Random Forest Regression model.
8. Predict car prices.
9. Evaluate the model using MAE, RMSE and R² score.
10. Visualize actual and predicted prices.

### Model Evaluation

The following metrics are used:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Visualization

An Actual vs Predicted Car Prices graph is generated to compare the model predictions with the actual selling prices.

### How to Run

Run the following command:

```bash
python car_price_prediction.py
```

### Learning Outcome

This project provides practical experience in regression, data preprocessing, categorical feature encoding, machine learning model training and evaluation using Scikit-learn.

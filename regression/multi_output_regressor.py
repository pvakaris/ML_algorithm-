import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# IMPORTANT
# For this script to work correctly, a file data.csv has to exist in the same directory as this script.
# The idea of the dataset is a list of house entries each with a set of attributes and a price.
# The algorithm is able to predict what parameters the hous eof a specific price should have.

# Load the dataset into a pandas DataFrame
dataframe = pd.read_csv("data.csv")

# Create a new dataset with only the price as a feature and the rest of the attributes as targets
X = dataframe[["price"]]
y = dataframe.drop("price", axis=1)
attribute_names = y.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multi-target regression model
regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
regr.fit(X_train, y_train)

# Given a price, predict the attributes of a house
def predict_attributes(price, model, targets):
    price = np.array([price]).reshape(1, -1)
    attributes = model.predict(price)
    return dict(zip(targets, attributes[0]))

# Example usage
price = 500000
attributes = predict_attributes(price, regr, attribute_names)
print("Attributes for a house with price $", price, ":")
for key, value in attributes.items():
    print("-", key, ":", value)
from random_forest import generate_forest, classify_forest, get_forest_accuracy
from cars_dataset import car_labels, car_data
from sklearn.model_selection import train_test_split

# Split the data to training and test data
X_train, X_test, y_train, y_test = train_test_split(car_data, car_labels, test_size=0.2, random_state=0)

forest = generate_forest(100, X_train, y_train)
print(get_forest_accuracy(forest, X_test, y_test))
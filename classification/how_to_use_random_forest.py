from random_forest import generate_forest, classify_forest, get_forest_accuracy
from cars_dataset import car_labels, car_data

forest = generate_forest(40, car_data, car_labels)
# print(get_forest_accuracy(forest, training_data, training_labels))
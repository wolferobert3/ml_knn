import math
import numpy as np
import csv
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

def scale_feature(feature):
    column_vector = np.array(feature)
    column_vector = np.subtract(column_vector, np.amin(column_vector))
    denominator = np.amax(column_vector) - np.amin(column_vector)
    if denominator == 0:
        column_vector = np.zeroes(len(column_vector))
        column_vector.shape = (len(column_vector), 1)
        return column_vector
    else:
        column_vector = np.divide(column_vector, denominator)
        return column_vector

def normalize(feature):
    column_vector = np.array(feature)
    no_zeroes = np.array(np.delete(column_vector, np.argwhere(column_vector == 0)))
    vector_mean = np.mean(no_zeroes)
    normalized = np.array(np.where(column_vector == 0, vector_mean, column_vector))
    return normalized

def matrix_euclidean(a1, v1):
    v1_arr = np.array([v1,]*len(a1))
    difference = np.subtract(v1_arr, a1)
    squared = np.power(difference, 2)
    summed = [sum(i) for i in squared]
    normed = [i**0.5 for i in summed]
    return np.array(normed)

def matrix_manhattan(a1, v1):
    v1_arr = np.array([v1,]*len(a1))
    difference = np.subtract(v1_arr, a1)
    absolute = np.absolute(difference)
    summed = [sum(i) for i in absolute]
    return np.array(summed)

def matrix_cosine(a1, v1):
    numerator = np.matmul(a1, v1)
    d1 = np.dot(v1, v1)**0.5
    d2 = np.einsum('ij,ij->i', a1, a1)**0.5
    denominator = np.multiply(d2, np.transpose(d1))
    quotient = np.divide(numerator, denominator)
    ones = np.array(np.ones(len(quotient)))
    cos = np.array(np.subtract(ones, quotient))
    return cos

def find_nearest(vector, labels, k):
    nearest_neighbors = []
    indices = []
    while len(nearest_neighbors) < k:
        idx = np.argmin(vector)
        indices.append(idx)
        nearest_neighbors.append(labels[idx])
        vector[idx] += np.amax(vector)
    for i in indices:
        vector[i] -= np.amax(vector)
    return max(set(nearest_neighbors), key = nearest_neighbors.count)

def make_predictions(k, training_data, training_labels, testing_data, dist_type):
    predictions = []

    if dist_type == 'euclidean':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_euclidean(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)
    
    if dist_type == 'manhattan':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_manhattan(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)

    if dist_type == 'cosine':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_cosine(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)

    return predictions

def test_binary_predictions(test_labels, test_predictions):
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    num_samples = len(test_predictions)
    for i in range(num_samples):
        true_positives = true_positives + 1 if test_labels[i] == 1 and test_predictions[i] == 1 else true_positives
        true_negatives = true_negatives + 1 if test_labels[i] == 0 and test_predictions[i] == 0 else true_negatives
        false_positives = false_positives + 1 if test_labels[i] == 0 and test_predictions[i] == 1 else false_positives
        false_negatives = false_negatives + 1 if test_labels[i] == 1 and test_predictions[i] == 0 else false_negatives

    tp_rate, tn_rate, fp_rate, fn_rate = true_positives / num_samples, true_negatives / num_samples, false_positives / num_samples, false_negatives / num_samples
    return [[true_positives, true_negatives, false_positives, false_negatives], [tp_rate, tn_rate, fp_rate, fn_rate]]

def confusion_matrix(test_labels, test_predictions, dims):
    conf_matrix = [[0 for i in range(0, dims)] for i in range(0, dims)]
    for i in range(len(test_labels)):
        conf_matrix[test_predictions[i]][test_labels[i]] += 1
    return conf_matrix

def percent_matrix(confusion_matrix, num_tests):
    percent_matrix = list(confusion_matrix)
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            percent_matrix[i][j] /= num_tests
    return percent_matrix

def simple_accuracy(labels, predictions):
    correct = 0
    for i in range(len(predictions)):
        correct = correct + 1 if labels[i] == predictions[i] else correct
    return correct / len(predictions)

### Pima Indians Diabetes Dataset Implementation
"""
t1 = time.time()

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path + '\\diabetes.csv', 'r', newline='') as dia_file:
    diabetes_file = list(csv.reader(dia_file))

diabetes_data = np.array(diabetes_file[1:], dtype=float)
diabetes_header = np.array(diabetes_file[0][:len(diabetes_file[0])-1])
diabetes_labels = np.array(diabetes_data[:, -1])
diabetes_data = np.delete(diabetes_data, len(diabetes_data[0])-1, axis=1)

diabetes_training = np.array(diabetes_data[int(len(diabetes_data)*0.2):])
dia_training_labels = np.array(diabetes_labels[int(len(diabetes_data)*0.2):])
diabetes_testing = np.array(diabetes_data[:int(len(diabetes_data)*0.2)])
dia_testing_labels = np.array(diabetes_labels[:int(len(diabetes_data)*0.2)])

for i in range(1, len(diabetes_training[0])):
    diabetes_training[:, i] = normalize(diabetes_training[:, i])
    diabetes_testing[:, i] = normalize(diabetes_testing[:, i])

for i in range(len(diabetes_training[0])):
    diabetes_training[:, i] = scale_feature(diabetes_training[:, i])
    diabetes_testing[:, i] = scale_feature(diabetes_testing[:, i])

predictions = make_predictions(7, diabetes_training, dia_training_labels, diabetes_testing, 'euclidean')
confusion_metrics = test_binary_predictions(dia_testing_labels, predictions)

wc_time = time.time() - t1

print(wc_time)
print(predictions)
print(confusion_metrics)

"""

### MNIST Implementation 
t2 = time.time()

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path + '\\train.csv', 'r', newline='') as mnist_train_file:
    mnist_train_data = list(csv.reader(mnist_train_file))

mnist_data = np.array(mnist_train_data[1:], dtype=int)

mnist_training = np.array(mnist_data[0:-1000, 1:])
mnist_training_labels = np.array(mnist_data[0:-1000, 0])
mnist_testing = np.array(mnist_data[-1000:, 1:])
mnist_testing_labels = np.array(mnist_data[-1000:, 0])

"""
mnist_training = np.array(mnist_data[:int(.9*(len(mnist_data))), 1:])
mnist_training_labels = np.array(mnist_data[:int(.9*(len(mnist_data))), 0])
mnist_testing = np.array(mnist_data[int(.9*(len(mnist_data))):, 1:])
mnist_testing_labels = np.array(mnist_data[int(.9*(len(mnist_data))):, 0])
"""

predictions_mnist = make_predictions(101, mnist_training, mnist_training_labels, mnist_testing, 'euclidean')
confusion_mnist = confusion_matrix(mnist_testing_labels, predictions_mnist, 10)
wc2 = time.time() - t2
print(wc2)
print(confusion_mnist)

with open(dir_path + '\\conf_matrix_k101.csv', 'w') as np_conf_matrix:
    np.savetxt(np_conf_matrix, np.array(confusion_mnist), delimiter=',')
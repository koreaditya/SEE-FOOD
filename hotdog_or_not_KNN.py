# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:56:33 2019

@author: Aditya Kore
"""
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import random


def calculate_cartesian_distance(distance1, distance2):
    """
    function calculates the cartesian distance

    :param distance1: values of training data point
    :type distance1:tuple
    :param distance2: values of testing data points
    :type distance2: tuple
    :return cartesian_distance: cartesian distance between training and testing data points
    :type: float
    """
    cartesian_product = 0

    for i in range(len(distance1)):
        cartesian_product = cartesian_product + (distance1[i] - distance2[i]) ** 2
    cartesian_distance = math.sqrt(cartesian_product)

    return cartesian_distance

def generate_dataset():
    """

    """
    data_dir = "train/"
    category = ["hot_dog", "not_hot_dog"]
    train_data = []
    hot_dog_count = 0
    for elements in category:
        path = os.path.join(data_dir, elements)
        for images in os.listdir(path):
            if elements == "hot_dog":
                hot_dog_count += 1
            image_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (50, 50))
            image_array = image_array.reshape(-1)
            train_data.append(image_array)

    data_dir = "test/"
    test_data = []
    for elements in category:
        path = os.path.join(data_dir, elements)
        for images in os.listdir(path):
            image_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (50, 50))
            image_array = image_array.reshape(-1)
            test_data.append(image_array)

    return train_data, test_data, hot_dog_count


class KNearestNeighbour(object):

    def __init__(self, train_data, test_data, k=3):
        """
        opens the file and calls calculate distance iteratively

        :param train_data: multi dimensional list of training data points
        :type train_data: list
        :param test_data: multi dimensional list of testing data points
        :type test_data: list
        :param k: value of k
        :type k: int
        """
        self.train_data = train_data
        self.test_data = test_data
        self.k = k
        self.file = open('output.txt', 'w')

    def calculate_distance(self, test_data_point):
        """
        calculate distances gets called in the function kNN, this function is called for every training dataset,
        this function calls calculate_cartesian_distances to calculate its distance to every training datapoint
        the distances are put in dictionary and sorted in ascending order, the key in dictionary is he distance
         and the values is the label.

        :param test_data_point: list of test data point
        :type test_data_point: list
        """
        test_data_point_distances = {}
        for elements in self.train_data:
            distance = calculate_cartesian_distance(elements[0], test_data_point)
            test_data_point_distances[distance] = elements
        test_data_point_distances = sorted(test_data_point_distances.items())
        self.classify(test_data_point_distances, test_data_point)

    def classify(self, test_data_point_distances, test_data_point):
        """
        in classify function we iterate through the test_data_point_distances which has distances in sorted order
        we add first k smallest distances to result dictionary
        in result dictionary we then classify the testing data point to the label with maximum value

        :param test_data_point_distances: distances between each testing data point and training data point.
        :type test_data_point_distances : list
        :param test_data_point: testing data point
        :type test_data_point: tuple
        """

        count = 1
        class_count = {}
        for elements in test_data_point_distances:
            if elements[-1][-1] not in class_count:
                class_count[elements[-1][-1]] = 1
            else:
                class_count[elements[-1][-1]] += 1
            if count == self.k:
                break
            count += 1
        max_count = 0
        for elements in class_count:
            if class_count[elements] > max_count:
                max_count = class_count[elements]
                datapoint_class = elements

        print("The data Point : "+ str(test_data_point)+ " is Classified to Label : " + datapoint_class)
        self.file.write("The data Point : "+ str(test_data_point)+ " is Classified to Label : " + datapoint_class+'\n')

    def kNN(self):
        self.file.write("k value is : " + str(self.k)+'\n')
        for elements in self.test_data:
            self.calculate_distance(elements)


def __main__():

    """
    the main function gets the dataset by calling the generate_dataset function
    """
    train_dataset, test_dataset, hot_dog_count = generate_dataset()

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(train_dataset)
    pca = PCA(n_components=3)
    pca.fit_transform(scaled_data_train)

    scaled_data_test = scaler.transform(test_dataset)
    pca.transform(scaled_data_test)
    y_pca = pca.transform(scaled_data_test)

    training_data = []
    print(hot_dog_count)
    for elements in train_dataset:
        if hot_dog_count > 0:
            training_data.append([elements, "H"])
            hot_dog_count -= 1
        else:
            training_data.append([elements, "N"])
    t = [1 for i in range(12)]
    for i in range(12):
        t.append(0)
    random.shuffle(training_data)
    T = y_pca
    print("train",training_data[0])
    print("t",T)
    obj = KNearestNeighbour(training_data, T, 3)
    obj.kNN()


if __name__ == __main__():
    __main__()

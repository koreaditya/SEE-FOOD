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


class LogisticRegression(object):
    """

    """
    def __init__(self, D, T, learning_rate):
        self.D = D
        self.T = T
        self.learning_rate = learning_rate

    def line_equation(self, b0, b1, b2, b3):
        """

        :param b0:
        :param b1:
        :param b2:
        :param b3:
        :return:
        """
        error = 0
        count_h=0
        count_nh=0
        for i in range(len(self.T)):
            x1=self.T[i][0]
            x2 = self.T[i][1]
            x3 = self.T[i][2]
            val = b0 + (b1 * x1) + (b2 * x2) + (b3 * x3)
            cl = 1/(1+math.exp(-val))
            cl=round(cl, 3)
            if cl > 0.490:
                if i < len(self.T)/2:
                    count_h += 1
                print(str(self.T[i])+" Clasified as"+': H')
            else:
                if i > len(self.T)/2:
                    count_nh += 1
                print(str(self.T[i]) + " Clasified as" + ': N')
        print(count_h)
        print(count_nh)

    def training_model(self):
        """

        """
        b0 = 0
        current_gradient_b1 = 0
        current_gradient_b2 = 0
        current_gradient_b3 = 0
        for i in range(10000):
            b0, current_gradient_b1, current_gradient_b2, current_gradient_b3 = self.gradient_descent(b0, current_gradient_b1, current_gradient_b2, current_gradient_b3)
        self.line_equation(b0, current_gradient_b1, current_gradient_b2, current_gradient_b3)

    def gradient_descent(self, current_b0, current_gradient_b1, current_gradient_b2, current_gradient_b3):
        """

        :param current_b0:
        :param current_gradient_b1:
        :param current_gradient_b2:
        :param current_gradient_b3:
        :return:
        """
        b0 = 0
        gradient_b1 = 0
        gradient_b2 = 0
        gradient_b3 = 0
        n = len(self.D)
        for i in range(n):
            x1 = self.D[i][0][0]
            x2 = self.D[i][0][1]
            x3 = self.D[i][0][2]
            if self.D[i][1] == 'H':
                y = 1
            else:
                y = -1

            b0 += -2*(y-(current_b0+(current_gradient_b1*x1)+(current_gradient_b2*x2)+(current_gradient_b3*x3)))
            gradient_b1 += -2 * x1 * (y-(current_b0+(current_gradient_b1*x1)+(current_gradient_b2*x2)+(current_gradient_b3*x3)))
            gradient_b2 += -2 * x2 * (y-(current_b0+(current_gradient_b1*x1)+(current_gradient_b2*x2)+(current_gradient_b3*x3)))
            gradient_b3 += -2 * x3 * (y-(current_b0+(current_gradient_b1*x1)+(current_gradient_b2*x2)+(current_gradient_b3*x3)))
        b0 = b0/n
        gradient_b1=gradient_b1/n
        gradient_b2 = gradient_b2 / n
        gradient_b3 = gradient_b3 / n
        new_b0 = current_b0 - (self.learning_rate * b0)
        new_gradient_b1 = current_gradient_b1 - (self.learning_rate * gradient_b1)
        new_gradient_b2 = current_gradient_b2 - (self.learning_rate * gradient_b2)
        new_gradient_b3 = current_gradient_b3 - (self.learning_rate * gradient_b3)
        return new_b0, new_gradient_b1, new_gradient_b2, new_gradient_b3


def generate_dataset():

    """

    """
    data_dir = "train/"
    category = ["hot_dog", "not_hot_dog"]
    train_data = []
    hot_dog_count = 0
    for elements in category:
        path=os.path.join(data_dir, elements)
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


def __main__():
    a, b, hot_dog_count=generate_dataset()

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(a)
    pca = PCA(n_components=3)
    pca.fit_transform(scaled_data_train)
    x_pca = pca.transform(scaled_data_train)

    scaled_data_test = scaler.transform(b)
    pca.transform(scaled_data_test)
    y_pca = pca.transform(scaled_data_test)

    D = []
    print(hot_dog_count)
    for elements in x_pca:
        if hot_dog_count > 0:
            D.append([elements, "H"])
            hot_dog_count -= 1
        else:
            D.append([elements, "N"])
    t=[1 for i in range(12)]
    for i in range(12):
        t.append(0)
    random.shuffle(D)
    T = y_pca
    obj = LogisticRegression(D, T, 0.00001)
    obj.training_model()


if __name__ == __main__():
    __main__()


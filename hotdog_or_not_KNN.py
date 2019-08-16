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

def calculate_cartesian_distance(distance1,distance2):
    cartesian_product=0
    for i in range(len(distance1)):
        cartesian_product=cartesian_product+(distance1[i]-distance2[i])**2
    cartesian_distance=math.sqrt(cartesian_product)
    return cartesian_distance

class KNearestNeighbour(object):
    #the function kNN spawns the testing datasets and calls calculate_distances
    #default value of k is 3
    def kNN(self,train_data,test_data,k=3):
        self.file=open('output.txt','w')
        self.k=k
        print("k value is : " + str(self.k))
        self.file.write("k value is : " + str(self.k)+'\n')
        self.train_data=train_data
        for elements in test_data:
            self.calculate_distance(elements)
    #calculate distances gets called in the function kNN, this function is called for every training dataset,
    #this function calls calculate_cartesian_distances to calculate its distance to every training datapoint
    #the distances are put in dictionary and sorted in ascending order, the key in dictionary is he distance and the values is the label.
    def calculate_distance(self,test_data_point):
        test_data_point_distances={}
        for elements in self.train_data:
            distance=calculate_cartesian_distance(elements[0],test_data_point)
            test_data_point_distances[distance]=elements
        test_data_point_distances=sorted(test_data_point_distances.items())
        self.clasify(test_data_point_distances,test_data_point)
    #in classify function we iterate through the test_data_point_distances which has distances in sorted order
    #we add first k smallest distances to result dictionary
    #in result dictionary we then classify the testing data point to the label with maximum value
    def clasify(self,test_data_point_distances,test_data_point):
        count=1
        class_count={}
        result={}
        for elements in test_data_point_distances:
            if elements[-1][-1] not in class_count:
                class_count[elements[-1][-1]]=1
            else:
                class_count[elements[-1][-1]]+=1
            if count==k:
                break
            count+=1
        max_count=0
        for elements in class_count:
            if class_count[elements]>max_count:
                max_count=class_count[elements]
                datapoint_class=elements
        print("The data Point : "+ str(test_data_point)+ " is Classified to Label : " + datapoint_class)
        self.file.write("The data Point : "+ str(test_data_point)+ " is Classified to Label : " + datapoint_class+'\n')

    
def generate_dataset():
    data_dir="train/"
    category=["hot_dog","not_hot_dog"]
    train_data=[]
    hot_dog_count=0
    for elements in category:
        path=os.path.join(data_dir,elements)
        for images in os.listdir(path):
            if elements=="hot_dog":
                hot_dog_count+=1
            image_array=cv2.imread(os.path.join(path,images),cv2.IMREAD_GRAYSCALE)
            image_array=cv2.resize(image_array,(50,50))
            image_array=image_array.reshape(-1)
            train_data.append(image_array)
            
    data_dir="test/"
    test_data=[]
    for elements in category:
        path=os.path.join(data_dir,elements)
        for images in os.listdir(path):
            image_array=cv2.imread(os.path.join(path,images),cv2.IMREAD_GRAYSCALE)
            image_array=cv2.resize(image_array,(50,50))
            image_array=image_array.reshape(-1)
            test_data.append(image_array)
            
    return train_data,test_data,hot_dog_count
a,b,hot_dog_count=generate_dataset()


scaler=StandardScaler()
scaled_data_train=scaler.fit_transform(a)
pca=PCA(n_components=3)
pca.fit_transform(scaled_data_train)
x_pca=pca.transform(scaled_data_train)

scaled_data_test=scaler.transform(b)
pca.transform(scaled_data_test)
y_pca=pca.transform(scaled_data_test)

D=[]
print(hot_dog_count)
for elements in a:
    if hot_dog_count>0:
        D.append([elements,"H"])
        hot_dog_count-=1
    else:
        D.append([elements,"N"])
t=[1 for i in range(12)]
for i in range(12):
    t.append(0)
random.shuffle(D)
T=y_pca
obj = KNearestNeighbour()
obj.kNN(D, T, 3)

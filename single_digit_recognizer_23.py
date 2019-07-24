#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:41:19 2019

@author: sarfaraz
"""

#from mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

print("Loading dataset...")
#mndata = MNIST("./data/")
#images, labels = mndata.load_training()
mndata = pd.read_csv("/home/sarfaraz/anaconda3/envs/opencv-forge/dataset/train.csv")
print(data)
mndata = pd.read_csv("/home/sarfaraz/anaconda3/envs/opencv-forge/dataset/train.csv").as_matrix()
#images, labels = mndata.load_training()
train_x = data[0:10000, 1:]
train_y = data[0:10000, 0]


clf = RandomForestClassifier(n_estimators=100)

# Train on the first 10000 images:
#train_x = images[:10000]
#train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = data[10000:11000,1:]
expected =data[10000:11000, 0]



print("Compute predictions")
p = clf.predict(test_x)
count = 0

for i in range(1, 1000):
    print(p[i],"\t", expected[i])
    
    #d = test_x[i]
    #d.shape = (28, 28)
    #pt.imshow(d, cmap ='gray')
    #print(clf.predict([xtest[190]]))
    #pt.show()

    if(p[i] == expected[i]):
        count = count+1
        

    

print("\nAccuracy :", count/1000 * 100)



#print("Accuracy: ", accuracy_score(expected, predicted))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 02:12:28 2019

@author: sarfaraz
"""
import pandas as pd
import matplotlib.pyplot as plt , matplotlib.image as mimg

from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv("/home/sarfaraz/anaconda3/envs/opencv-forge/dataset/train.csv")
images = labeled_images.iloc[0:500, 1:]
labels = labeled_images.iloc[0:500, :1]

train_images , test_images, train_labels , test_labels = train_test_split(images, labels, train_size= 0.8, random_state = 0)



i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))

print("\n")

test_images[test_images>0]=1
train_images[train_images>0]=1

img = train_images.iloc[i].as_matrix().reshape((28,28))
#plt.imshow(img,cmap='binary')
#plt.title(train_labels.iloc[i])


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))





'''
#print(test_labels)
#print(type(test_labels))
#clf = svm.SVC()

#clf.fit(train_images, train_labels.values.ravel())


#p = clf.predict(train_images)
#print(p)
#print(test_labels)
#count = 0

#for i in range(0, 100):
 #   print(p[i],"\t",test_labels[i].values.ravel())
    #d = test_images[i]
    #d.shape = (28, 28)
    #pt.imshow(d, cmap ='gray')
    #print(clf.predict([xtest[190]]))
    #pt.show()

  #  if(p[i] == test_labels[i]):
   #     count = count+1
        

    

#print("\nAccuracy :", count/100 * 100)'''
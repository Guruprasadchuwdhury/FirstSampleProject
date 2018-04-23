# Python Basic

# String
#-------------------------------
print ('hello \ World')
print (len("hello World"))
s = 'hello world'
print (s)
print (s.capitalize())
print (s.upper())
print (s.replace('l','(ell)'))
print (s.islower())
print (s[1])


# List
#---------------------------
sx = [1, 2, 3, 4, 5]
print (sx)
print (sx[2:4])
print (sx[-2])
sx[2:4] = [7, 8, 9, 0]
print (sx)
num = list(range(7))
print ('Num is %s' % num)
print (num[-5:-2])
num[2] = 'foo'
num[4] = 'bar'
num.append('mar')
#num[8] = 'test'
print (num)
last = num.pop()
print('last element was \'%s\' and num is %s' % (last, num))



# Loop
#-----------------------------------
animals = ['cat', 'dog', 'monkey']
animals.append('cow')
animals.append('rat')
print (animals)
for animal in animals:
  print (animal)
myList = []
myList2 = []
for (idx, animal) in enumerate(animals):
  print ('#%d : %s' % (idx+1, animal))
  myList.append('#%d : %s' % (idx+1, animal))
print (myList)
#myList2 = [(idx+1, animal) for idx+1, animal in enumerate(animals)]
print (myList2)


# Dictionaries
#-----------------------------------
d = {'cat': 'cute', 'dog': 'furry'}
print (d['dog'])
print ('cat' not in d)
d['fish'] = 'wet'
print (d)
print (d.get('fish', 'Not Found'))
del d['fish']
print (d.get('fish', 'Not Found'))

for animal in d:
  print ('%s behavior is %s' % (animal, d[animal]))
  
nums = list(range(10))
dec = {num : num**3 for num in nums}
print (dec)


# Sets
#--------------------------------------
animals = {'cat', 'dog'}
animals.add('fish')
print (len(animals), animals)
animals.remove('cat')
if ('cat' in animals):
  animals.remove('cat')
print (len(animals), animals)
animal = animals.pop()
print (len(animal), animal)
animals.add('lion')
animals.add('lion')
animals.add('Tiger')      # Why Tiger ans lion appears in first in sets ?
print (len(animals), animals)
animals.add('lion')
print (len(animals), animals)


# Touples
#-----------------------------------
t = (7,7**2)
print (t)
dic = {(x,x**2): x+10 for x in range(10)}
print (dic)
if (t in dic):
    print (dic[t])
​
tp = (5, 5+21)
print(dic.get(tp, 'Not Listed'))
​
t1 = (7,7**2, 7**3, 'test', t, dic)
print (t1)
t1 = (7,7**2, 'test', dic)
print (t1)



# Functions
#-----------------------------------

def hello(name, loud = False):
  if (loud == True):
    print ('HELLO %s' % name.upper())
  else:
    print ('Hello %s' % name)
​
hello('Bob', False)
hello ('Guru', loud = True)
​
#Recursion
def rec (num):
  if ((num == 1) or (num == 0)):
      return 1
  else:
      return (num * rec(num-1))
​
def itr(num):
  res = 1
  for i in range(num):
    if (num+1 == 0 or num+1 == 1):
      res*= 1
    else:
      res *= i+1
  return res
    
    
num = input("Enter the number: ")
print ('Recursive Factorial is : %d' % rec (int(num)))
print ('Iterative Factorial is : %d' % itr (int(num)))



# NumPy
#-----------------------------------
import numpy as npy
​
x = npy.array([1, 2, 3])
print(x.shape)
​
print(x[0], x[1], x[2])
x[0] = 5
print (x[0])
​
y = np.array([[1,2,3],[4,5,6]])
print (y.shape)
print(y[0, 0], y[0, 1], y[1, 0]) 
​
t = (y.shape)
print (type(t))
​
a = np.zeros((2,2))
print (a)
​
b = np.ones((2,3))
print (b)
​
c = np.full((2,2),-1)
print (c)
​
e = np.random.random((2,2))
print(e)


# Array Indexing
#-----------------------------------
import numpy as x
​
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (a)
​
print ('-------------------------------------------')
​
row_r1 = a[1, :]    # Rank 1 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
​
# [row, column] ---  a[0:2, 1:3] means 
# display 0-2 row and 1-2 col
row_r2 = a[0:2, 1:3]  # Rank 2 view of the second row of a
print(row_r2, row_r2.shape)
​
print ('-------------------------------------------')
b = np.array ([[[1,2,3,4],[5,6,7,8], [9,10,11,12]], [[13,14,15,16],[17,18,19,20], [21,22,23,24]]])
myRank1 = b[0:2, 0:3, 1:3]
print (myRank1, myRank1.shape)



# Array Math
#-----------------------------------
import numpy as n
​
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print (x, y)
print (np.add(x, y))
print ('------------------------------------------')
print (np.subtract(x, y))
print (x * y)
print ('------------------------------------------')
print(np.sqrt(x))
​


# Dot Function
#-----------------------------------
import numpy as np
​
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
​
v = np.array([9,10])
w = np.array([11, 12])
​
# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))
​
print(x.dot(v))
print(np.dot(x, v))
​
print(x.dot(y))
print(np.dot(x, y))
​
print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"



# Matplotlib
#-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
​
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
print (x)
y = np.sin(x)
print (y)
# Plot the points using matplotlib
plt.plot(x, y)
#plt.show()  # You must call plt.show() to make graphics appear.
​
a = [2.5, 1, 0,   1, 2.5, 4, 5,   4, 2.5]
b = [0,   1, 2.5, 4, 5,   4, 2.5, 1, 0]
​
plt.plot(a, b)
#plt.show()
​
m = np.arange(0, 20, 1)
n = np.arange(0, 2, 0.1)
​
plt.plot(m,n)
plt.show()
​


# Images
#-----------------------------------
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import os
os.getcwd()
​
​
img = imread('cat.jpg')
print (img.dtype, img.shape)
​
# Show the original image
plt.subplot(1, 3, 1)
plt.imshow(img)
​
# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (223, 140, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0, 1]
img_tinted1 = img * [0, 1, 1]
​
# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))
​
# Show the tinted image
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(img_tinted))
​
# Write the tinted image back to disk
#imsave('cat_tinted.jpg', img_tinted)
​
plt.subplot(1, 3, 3)
#img1 = imread('programming-languages.jpeg')
#img1 = imresize(img1, (200, 300))
plt.imshow(np.uint8(img_tinted1))
plt.show()
​
​
# Linux Commands
#-----------------------------------------------
#!uname -a
#!lscpu
#!cat /proc/meminfo
!date
!pwd
#!df -h
#!mkdir guru
#!cp cat.jpg catttt.jpeg
#!rm catttt.jpeg
#!mkdir guru
#!rm -rf guru
!ls
!gcc test.c
!./a.out



# File Upload
#-----------------------------------------------
from google.colab import files
​
uploaded = files.upload()
​
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

	  
	  
	  
# File Download
#-----------------------------------------------
from google.colab import files
​
with open('test.c', 'w') as f:
  f.write('#include <stdio.h>\n int main() { \nprintf("Hello World !");\nreturn 0;\n}')
​
files.download('cat.jpg')




# Pratice
#-----------------------------------------------
!pip install -q keras #if you are on Google Colab
import keras
import numpy as np
​
from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D
from keras.utils import np_utils
​
from keras.datasets import mnist
​
# Load pre-shuffled MNIST data into train and test sets
​
(X_train, y_train), (X_test, y_test) = mnist.load_data()
​
print (X_train.shape)
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(X_train[0])
​
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
​
y_train[:10]
​
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
​
Y_train[:10]
​
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(10, 26))
model.add(Flatten())
model.add(Activation('softmax'))
​
model.summary()
​
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
​
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
​
score = model.evaluate(X_test, Y_test, verbose=0)
​
print(score)
​
y_pred = model.predict(X_test)
​
print(y_pred[:9])
print(y_test[:9])
[ ]

import cv2
​
image = cv2.imread("cat.jpg")
edges = cv2.Canny(image, 10, 20)
​
cv2.imwrite("content/rf4", edges)
​
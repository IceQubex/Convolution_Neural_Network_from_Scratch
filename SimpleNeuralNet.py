import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import os
import random

#simple image scaling to (nR x nC) size
def scale(im, nR, nC):
  nR0 = len(im)     #source number of rows
  nC0 = len(im[0])  #source number of columns
  return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]

#defining activation functions
def softmax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))
def ReLU(x):
    return np.maximum(np.zeros(x.shape), x)
def leaky_ReLU(x):
    return np.maximum(0.01*x,x)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#defining derivative of activation functions
def der_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0
def der_leaky_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0.01
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def der_tanh(x):
    return (1-(np.power(tanh(x),2)))

# define the neural network class and functions
class NeuralNetwork:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        self.num_of_classes = np.max(self.train_label) + 1
        self.num_of_input = len(self.train_data[0].flatten())
        self.bias = np.zeros(self.num_of_classes)
        self.weights = np.zeros((self.num_of_classes, self.num_of_input))
        self.output = np.zeros(self.num_of_classes)
        self.onehot = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

    def train(self, max_iterations):
        self.max_iterations = max_iterations
        for iterations in range(self.max_iterations):
            for i in range(len(train_data)):
                for j in range(len(self.output)):
                    self.output[j] = np.dot(self.weights[j],self.train_data[i].flatten()) + self.bias[j]
                self.output = softmax(self.output)
                loss = -(np.log(self.output[train_label[i]]+0.0000000000000001))
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_w = np.zeros(self.weights.shape)
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] = self.output[k] - self.onehot[train_label[i]][k]
                    for l in range(self.num_of_input):
                        self.d_loss_d_w[k][l] = 0.025 * self.d_loss_d_output[k] * train_data[i].flatten()[l]
                self.weights -= self.d_loss_d_w
                print(str(i+1)+" images done!")
            print(str(iterations+1) + " iterations Done!")

    def evaluate(self, test_data, test_label):
        correct = 0
        for i in range(len(test_data)):
            for j in range(len(self.output)):
                self.output[j] = np.dot(self.weights[j],train_data[i].flatten()) + self.bias[j]
            self.output = softmax(self.output)
            if np.argmax(self.output) == test_label[i]:
                correct += 1
        print("The accuracy of the predictions is " + str((correct/len(test_data))*100) + "%.")

    def predict(self, test_data, test_label):
        self.test_data = test_data
        self.test_label = test_label
        for m in range(len(self.output)):
            self.output[m] = np.dot(self.weights[m],self.test_data.flatten()) + self.bias[m]
        self.output = softmax(self.output)
        print("The data is predicted to be in class " + str(np.argmax(self.output))+".")

# Read the train and test data and labels
train_data = []
train_label = []
for i in range(len(os.listdir("Pics"))):
    print("Hand gesture "+os.listdir("Pics")[i]+" is assigned to class " + str(i)+".")
    for j in range(len(os.listdir("Pics/"+os.listdir("Pics")[i]))):
        img = imread("Pics/"+os.listdir("Pics")[i]+"/Pic (" + str(j+1) + ").jpg")
        img1 = scale(img,90,160) #to downscale the images
        img1 = np.array(img1)
        train_data.append(img1)
        train_label.append(i)
temp = list(zip(train_data, train_label))
random.shuffle(temp)
temp_train = temp[0:80]
temp_test = temp[80:]
train_data, train_label = zip(*temp_train)
test_data, test_label = zip(*temp_test)
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

#create the NeuralNetwork instance
test = NeuralNetwork(train_data, train_label)
test.train(1)
print("Training is done!")
plt.imshow(test_data[1])
plt.show()
test.predict(test_data[1],test_label[1])

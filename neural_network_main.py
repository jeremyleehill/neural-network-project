#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:31:37 2021

3-Layer Neural Network trained on MNIST dataset

@author: Jeremy Hill
(c) Jeremy Hill, 2021
licence is GPLv2
"""

from neural_Network import NeuralNetwork

import numpy as np

def main():
    # number of input, hidden, and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    
    # learning rate is 0.1
    learning_rate = 0.1
    
    # create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    # train the neural network
    
    # epochs is the number of times the training data set is used for training
    epochs = 5
    
    for e in range(epochs):
        # go through all the records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            
            # create the target output values (all 0.01, except the desired
            # label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            
            n.train(inputs, targets)
            
            pass
        pass
        
    # test the neural network    
    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    # scorecard for how well the network performs, initially empty
    scorecard = []
    
    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        
        # correct answer is first value
        correct_label = int(all_values[0])
        #print(correct_label, "correct label")
        
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        # query the network
        outputs = n.query(inputs)
        
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        #print(label, "network's answer")
        
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
            
        else:
            # network's answer doesn't match correct answer,
            # add 0 to scorecard
            scorecard.append(0)
            
            pass
        
        pass
                
    #print(scorecard)
        
    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
    

    
if __name__ == '__main__':
    
    main()
    

    
    
            
    
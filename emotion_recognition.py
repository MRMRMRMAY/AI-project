from __future__ import division, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import random
import sys
import os

class EmotionRecognition:

    def __init__(self):
        self.dataset = DatasetLoader()

    def build_network(self):
        # Smaller 'AlexNet'
        # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
        print('[+] Building CNN')
        '''
        # Why 3 hidden layers?
        # 1986: Backpropagation - Usually more than 3 hidden layer is not helpful
        '''

        '''
        [-]input_data()
        #This layer is use for inputting data to a network.
        #List of int, to create a new placeholder
        # shape = [batch, height, width, in_channels]
        '''
        self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1]) # add data whose shape is [None,48, 48 ,1] into an 'input_data' layer

        '''
        [-]conv_2d
        #arg1 - incoming: [batch, height, width, in_channels]
        #arg2 - nb_filter: The number of convolution filters
        #arg3 - filter_size : Size of filters
        '''
        self.network = conv_2d(self.network, 64, 5, activation='relu') # 1st layer
        #self.network = local_response_normalization(self.network) #

        '''
        [-]max pooling 2D
        
        # arg1 - incoming:
        # arg2 - kernel_size: Pooling kernel size
        # arg3 - strides : stides of conv operation  e.g,(0,0)->(0,2)->(0,4)
        '''
        self.network = max_pool_2d(self.network, 3, strides=2) # pool
        self.network = conv_2d(self.network, 64, 5, activation='relu') # 2nd layer
        self.network = max_pool_2d(self.network, 3, strides=2) # pool
        self.network = conv_2d(self.network, 128, 4, activation='relu') # 3rd layer
        '''
        [-]Dropout
        reference: tflearn.org/layers/core/#dropout
        Introduction:
        #Outputs the input element scaled up by 1/keep_prob. The scaling is so that the expected sum is unchanged
        #By default, each element is kept or dropped independently. If noise_shape is specified, it must be broadcastable to the shape of x, and only dimensions with noise_shape[i] == shape(x)[i] will make
        independent decisions. For example, if shape(x) = [k, l, m, n] and noise_shape = [k, 1, 1, n], each batch and channel component will be kept independently and each row and column will be kept or not kept together
        
        #arg1 - incoming: []
        #arg2 - keep_prob: A float representing the probability that each element is kept
        '''
        self.network = dropout(self.network, 0.3) # final: output layer

        '''
        [-]fully_connected
        return : 2D Tensor[samples, n_units]
        
        arg1 - incoming: 2+D Tensor []
        arg2 - n_units: the # of units for this layer
        '''
        self.network = fully_connected(self.network, 3072, activation='relu') # A fully connected layer
        self.network = fully_connected(
            self.network, len(EMOTIONS), activation='softmax') # A fully connected layer
        '''
        [-]regression
        To apply a regreesion to the provided input.
        # optimizer: Optimizer to use
        # loss: Loss function used by this layer optimizer  
        '''
        self.network = regression(
            self.network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )# conput loss and optimizer

        '''
        Deep Neural Network Model
        # network: NN to be used
        # checkpoint_path : the path to store model file
        # max_checkpoint: Maximum amount of checkpoints
        # tensorboard_verbose: Summary verbose level, it accepts different levels of tensorboard logs.
        '''
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path=SAVE_DIRECTORY + '/emotion_recognition',
            max_checkpoints=1,
            tensorboard_verbose=2
        ) #model max_checkpoints = 1: save only one model file.
        self.load_model()

    def load_saved_dataset(self):
        self.dataset.load_from_save()
        print('[+] Dataset found and loaded')
    '''training method'''
    def start_training(self):
        self.load_saved_dataset()
        self.build_network()
        if self.dataset is None:
            self.load_saved_dataset()
        # Training
        print('[+] Training network')
        ''' 
        method fit() : train model, feeding X_inputs and Y_targets to the network.
        # arg1: training data ---- X_inputs; 
        # arg2: training label --- Y_targets; 
        # validation_set : represents data used for validation
        # n_epoch = 10: The # of epoch to run
        # shuffle: overrides all network estimators 'shuffle' by True
        # show_metric: Display or not accuracy at every step
        # snapshot_step: to save the model every X steps
        # snapshot_epoch: to save the model at the end of every epoch
        # run_ip: give a name for this run
        '''
        self.model.fit(
            self.dataset.images, self.dataset.labels,
            validation_set=(self.dataset.images_test,
                            self.dataset.labels_test),
            n_epoch=10,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=1000,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )
    '''Model prediction for given input data'''
    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return self.model.predict(image) # return the predicted probabilities
    '''save the model'''
    def save_model(self):
        self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
        print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)
    '''load the model trained'''
    def load_model(self):
        if os.path.exists(SAVE_DIRECTORY):
            print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
            self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
        # if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
        #     print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
        #     self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))


def show_usage():
    # I din't want to have more dependecies
    print('[!] Usage: python emotion_recognition.py') #python emotion_recognition.py train
    print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
    print('\t emotion_recognition.py poc \t Launch the proof of concept')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()
    network = EmotionRecognition()
    if sys.argv[1] == 'train':
        network.start_training()
        network.save_model()
    else:
        show_usage()

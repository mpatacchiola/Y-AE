#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Convolutional Autoencoder.
#The decoder is obtained via upsampling and convolution.
#The class is an implementation of an Autoencoder that can be used
#to train a model on the SVHN dataset. The class is flexible enough and
#can be readapted to other datasets. Methods such as save() and load()
#allow the user to save and restore the network. A log file is locally stored
#and can be used to visualize the training from tensorboard.

#TENSORBOARD: tensorboard --logdir=./ --host=localhost --port=8088

import tensorflow as tf
import numpy as np
import datetime
from time import gmtime, strftime
import os
#import scipy.io as sio
from tensorflow.python.lib.io import file_io
from io import BytesIO

class LeNet:
    def __init__(self, batch_size, tot_labels=10, channels=1, conv_filters=64, ksize=(3,3), start_iteration=0, dir_header="./"):
        '''Init method
        @param sess (tf.Session) the current session
        '''
        #Resource exhausted: OOM when allocating tensor with shape[128,64,32,32]
        self.dir_header = dir_header
        self.start_iteration = start_iteration
        self.channels = channels
        activation_function =  None #tf.nn.leaky_relu
        weight_initializer = None #tf.truncated_normal_initializer(mean=0.0, stddev=0.03) #None
        regularizer = None #tf.contrib.layers.l2_regularizer(0.01) #None

        with tf.variable_scope("Input", reuse=False):
            # x is (32 x 32 x 3)
            self.x = tf.placeholder(tf.float32, [batch_size, 32, 32, self.channels]) #Input
            self.labels_placeholder = tf.placeholder(tf.int64,[batch_size, tot_labels])

        ##ROOT
        with tf.variable_scope("Net", reuse=False):
            #Conv-1 -> (16, 16, conv_filters)
            conv_1 = tf.layers.conv2d(inputs=self.x, filters=conv_filters, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_1")
            conv_1 = tf.layers.batch_normalization(conv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            conv_1 = tf.nn.leaky_relu(conv_1, name="relu_1")
            ##Conv-2 -> (8, 8, conv_filters*2)
            conv_2 = tf.layers.conv2d(inputs=conv_1, filters=conv_filters*2, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_2")
            conv_2 = tf.layers.batch_normalization(conv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            conv_2 = tf.nn.leaky_relu(conv_2, name="relu_2")
            ##Conv-3 -> (4, 4, conv_filters*4)
            conv_3 = tf.layers.conv2d(inputs=conv_2, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_3")
            conv_3 = tf.layers.batch_normalization(conv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            conv_3 = tf.nn.leaky_relu(conv_3, name="relu_3")
            ##Conv-4 -> (2, 2, conv_filters*8)
            conv_4 = tf.layers.conv2d(inputs=conv_3, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_4")
            conv_4 = tf.layers.batch_normalization(conv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            conv_4 = tf.nn.leaky_relu(conv_4, name="relu_4")
            ##Conv-5 OUTPUT
            conv_5 = tf.layers.conv2d(inputs=conv_4, filters=tot_labels, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_5")
            conv_5 = tf.layers.batch_normalization(conv_5, axis=-1, momentum=0.99, epsilon=0.001, name="norm_5")
            conv_5 = tf.nn.leaky_relu(conv_5, name="relu_5")
            conv_5_squeezed = tf.squeeze(conv_5)
            hidden = tf.layers.dense(conv_5_squeezed, units=64, activation=tf.nn.leaky_relu)
            self.output_logits = tf.layers.dense(hidden, units=tot_labels, activation=None)
            self.output_argmax = tf.argmax(self.output_logits, axis=1, name="output_argmax")
            self.output = tf.nn.softmax(self.output_logits, name="output")

        #Train operations
        with tf.variable_scope("Training"):
            ##Loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_placeholder, logits=self.output_logits)
            self.accuracy = tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(self.labels_placeholder,axis=1), self.output_argmax), dtype=tf.float32))

            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)

            self.tf_saver = tf.train.Saver()
            self.train_iteration = start_iteration
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.scalar("loss", self.loss, family="_loss_main")
            tf.summary.scalar("accuracy", self.accuracy, family="accuracies")
                                    
    def init_summary(self, sess):
        summary_id = strftime("%H%M%S_%d%m%Y", gmtime())
        summary_folder = self.dir_header + '/log/' + summary_id + '_iter_' + str(self.start_iteration)
        self.tf_summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
        self.summaries = tf.summary.merge_all() #merge all the previous summaries
        
    def forward(self, sess, input_feature):
        '''Feed-forward pass in the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (np.array) the output of the autoencoder
        '''
        output = sess.run([self.output], feed_dict={self.x: input_feature})
        return output

    def test(self, sess, input_features, input_labels):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        loss, accuracy  = sess.run([self.loss, self.accuracy], feed_dict={self.x: input_features, self.labels_placeholder: input_labels})
        return loss, accuracy

    def train(self, sess, input_features, input_labels, learning_rate, iteration, summary_rate=250):
        '''Single step training of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        _, loss, summ = sess.run([self.train_op, self.loss, self.summaries], feed_dict={self.x: input_features, self.labels_placeholder: input_labels, self.learning_rate: learning_rate})
        if(self.train_iteration % summary_rate == 0):
            self.tf_summary_writer.add_summary(summ, global_step=self.train_iteration)
            self.tf_summary_writer.flush()
        self.train_iteration = iteration
        return loss


    def save(self, sess, verbose=True):
        '''Save the model
        @param sess (tf.Session) the current session
        @param verbose (bool) if True print information on terminal
        '''
        if not os.path.exists(self.dir_header + "/model/"):
            os.makedirs(self.dir_header + "/model/")
        model_id = strftime("%H%M%S_%d%m%Y", gmtime())
        model_folder = self.dir_header + "/model/" + model_id + "_" + str(self.train_iteration) + "/model.ckpt"
        if(verbose): print("Saving networks in: " + str(model_folder))
        save_path = self.tf_saver.save(sess, model_folder)

    def load(self, sess, file_path, verbose=True):
        '''Load a model
        NOTE: when loading a model the method tf.global_variables_initializer()
        must not be called otherwise the variables are set to random values
        @param sess (tf.Session) the current session
        @param verbose (bool) if True print information on terminal
        '''
        if(verbose): print("Loading networks from: " + str(file_path))
        save_path = self.tf_saver.restore(sess, file_path)
        if(verbose): print("Done!")

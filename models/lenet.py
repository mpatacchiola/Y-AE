#The MIT License (MIT)
#Copyright (c) 2019 anonymous authors
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os

class LeNet:
    def __init__(self, batch_size, channels=1, conv_filters=8, tot_labels=10, ksize=(5,5), start_iteration=0, dir_header="./", wdecay=0.0):
        '''Init method
        @param sess (tf.Session) the current session
        @param conv_filters_* (int) the number of filters in the convolutional layers
        @param tot_labels (int) the number of units in the output layer
        @param ksize the size of the convolutional kernels
        @param wdecay the value used for the weight decay (default=0; not used)
        '''
        self.dir_header = dir_header
        self.start_iteration = start_iteration
        self.channels = channels
        weight_initializer = None #Tf default init
        weight_initializer_implicit = None
        bias_initializer_implicit = None
        if(wdecay > 0.0):
            regularizer = tf.contrib.layers.l2_regularizer(wdecay) #0.00001 is a good choice
        else:
            regularizer = None

        with tf.variable_scope("Input", reuse=False):
            # x is (32 x 32 x 3)
            self.x = tf.placeholder(tf.float32, [batch_size, 32, 32, self.channels]) #Input
            self.labels_placeholder = tf.placeholder(tf.int64,[batch_size])

        with tf.variable_scope("Network", reuse=False):
            c1 = tf.layers.conv2d(inputs=self.x, filters=conv_filters, kernel_size=ksize, 
                                  padding="valid", activation=tf.nn.relu, kernel_regularizer=regularizer)
            s2 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=2, padding="valid")
            c3 = tf.layers.conv2d(inputs=s2, filters=conv_filters*2, kernel_size=ksize, 
                                  padding="valid", activation=tf.nn.relu, kernel_regularizer=regularizer) 
            s4 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2)
            s4_flat = tf.reshape(s4, [-1, 5 * 5 * (conv_filters*2)])   
            c5 = tf.layers.dense(inputs=s4_flat, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)   
            f6 = tf.layers.dense(inputs=c5, units=64, kernel_regularizer=regularizer)
            self.logits = tf.layers.dense(inputs=f6, units=tot_labels, kernel_regularizer=regularizer)
            self.output_argmax = tf.argmax(self.logits, axis=1)
            
        #Train operations
        with tf.variable_scope("Training"):
            ##Global Loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(indices=self.labels_placeholder, depth=tot_labels), logits=self.logits)
            self.accuracy = tf.reduce_mean(tf.cast(tf.math.equal(self.labels_placeholder, self.output_argmax), dtype=tf.float32))
            ##Train ops
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            ##Saver
            self.tf_saver = tf.train.Saver()
            self.train_iteration = start_iteration
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.scalar("loss", self.loss, family="_loss_main")
            tf.summary.scalar("accuracy", self.accuracy, family="accuracy")

    def init_summary(self, sess):
        '''Init the summary folder with current time and date
        @param sess (tf.Session) the current session
        '''
        summary_id = strftime("%H%M%S_%d%m%Y", gmtime())
        summary_folder = self.dir_header + '/log/' + summary_id + '_iter_' + str(self.start_iteration)
        self.tf_summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
        self.summaries = tf.summary.merge_all() #merge all the previous summaries
        
    def forward(self, sess, input_feature):
        '''Feed-forward pass in the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix of features
        @return (np.array) the output of the autoencoder (reconstruction)
        '''
        output = sess.run([self.output], feed_dict={self.x: input_feature})
        return output
 
    def test(self, sess, input_features, input_labels):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @param lambda_e explicit mixing coefficient
        @param lambda_i implicit mixing coefficient
        @return (float) the losses: loss, loss_r, loss_c, acc_c, loss_e, loss_i
        '''
        loss  = sess.run([self.loss, self.accuracy], feed_dict={self.x: input_features, self.labels_placeholder: input_labels})
        return loss

    def train(self, sess, input_features, input_labels, learning_rate, iteration, summary_rate=250):
        '''Single step training of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix of features
        @param input_labels (np.array) array of labels
        @param learning_rate
        @param lambda_e explicit mixing coefficient
        @param lambda_i implicit mixing coefficient
        @param iteration the current iteration (used for the summary index)
        @param summary_rate summary written at this rate (iterations)
        @return (float) the global loss
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
        if(verbose): print("Saving network in: " + str(model_folder))
        save_path = self.tf_saver.save(sess, model_folder)

    def load(self, sess, file_path, verbose=True):
        '''Load a model
        NOTE: when loading a model the method tf.global_variables_initializer()
        must not be called otherwise the variables are set to random values
        @param sess (tf.Session) the current session
        @param path to the model folder, note that the path should end with '/model.ckpt' 
            even though this object does not exists in the path
        @param verbose (bool) if True print information on terminal
        '''
        if(verbose): print("Loading network from: " + str(file_path))
        save_path = self.tf_saver.restore(sess, file_path)
        if(verbose): print("Done!")

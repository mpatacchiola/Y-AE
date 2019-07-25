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

class Autoencoder:
    def __init__(self, batch_size, channels=1, conv_filters=8, style_size=32, content_size=10, ksize=(3,3), start_iteration=0, dir_header="./", wdecay=0.0):
        '''Init method
        @param sess (tf.Session) the current session
        @param conv_filters_* (int) the number of filters in the convolutional layers
        @param code_size (int) the number of units in the code layer
        @param gradient_clip (bool) applies gradient clipping on the gradient vector
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

        with tf.variable_scope("Encoder", reuse=False):
            #Conv-1 -> (16, 16, conv_filters)
            conv_1 = tf.layers.conv2d(inputs=self.x, filters=conv_filters, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_1")
            conv_1 = tf.layers.batch_normalization(conv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            conv_1 = tf.nn.leaky_relu(conv_1, name="relu_1")
            ##Conv-2 -> (8, 8, conv_filters*2)
            conv_2 = tf.layers.conv2d(inputs=conv_1, filters=conv_filters*2, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_2")
            conv_2 = tf.layers.batch_normalization(conv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            conv_2 = tf.nn.leaky_relu(conv_2, name="relu_2")
            ##Conv-3 -> (4, 4, conv_filters*4)
            conv_3 = tf.layers.conv2d(inputs=conv_2, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_3")
            conv_3 = tf.layers.batch_normalization(conv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            conv_3 = tf.nn.leaky_relu(conv_3, name="relu_3")
            ##Conv-4 -> (2, 2, conv_filters*4)
            conv_4 = tf.layers.conv2d(inputs=conv_3, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_4")
            conv_4 = tf.layers.batch_normalization(conv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            conv_4 = tf.nn.leaky_relu(conv_4, name="relu_4")
            ##Conv-7 STYLE
            conv_5 = tf.layers.conv2d(inputs=conv_4, filters=style_size, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer_implicit, bias_initializer=bias_initializer_implicit, name="conv_5")
            self.code_style = tf.nn.sigmoid(tf.squeeze(conv_5) , name="code_style")
            ##Conv-8 CONTENT
            conv_6 = tf.layers.conv2d(inputs=conv_4, filters=content_size, strides=(2,2), kernel_size=ksize, padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_6")
            self.code_content_logits = tf.squeeze(conv_6) 
            self.code_content = tf.nn.softmax(self.code_content_logits, name="code_content")

        with tf.variable_scope("Decoder", reuse=False):
            self.code_content_deterministic = tf.one_hot(indices=self.labels_placeholder, depth=content_size)
            code = tf.concat([self.code_style, self.code_content_deterministic], axis=1)
            ##Transpose-Convolution-1 -> (2, 2, conv_filters*4)
            code_reshaped = tf.reshape(code, [batch_size, 1, 1, style_size+content_size])
            deconv_1 = tf.layers.conv2d_transpose(code_reshaped, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_1")
            deconv_1 = tf.layers.batch_normalization(deconv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            deconv_1 = tf.nn.leaky_relu(deconv_1, name="relu_1")
            ##Transpose-Convolution-2 -> (4, 4, conv_filters*4)
            deconv_2 = tf.layers.conv2d_transpose(deconv_1, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_2")
            deconv_2 = tf.layers.batch_normalization(deconv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            deconv_2 = tf.nn.leaky_relu(deconv_2, name="relu_2")
            ##Transpose-Convolution-3 -> (8, 8, conv_filters*4)
            deconv_3 = tf.layers.conv2d_transpose(deconv_2, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_3")
            deconv_3 = tf.layers.batch_normalization(deconv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            deconv_3 = tf.nn.leaky_relu(deconv_3, name="relu_3")
            ##Transpose-Convolution-4 -> (16, 16, conv_filters*4)
            deconv_4 = tf.layers.conv2d_transpose(deconv_3, filters=conv_filters*2, kernel_size=ksize, strides=(2,2), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_4")
            deconv_4 = tf.layers.batch_normalization(deconv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            deconv_4 = tf.nn.leaky_relu(deconv_4, name="relu_4")
            ##Transpose-Convolution-5 -> (32, 32, conv_filters*2)
            deconv_5 = tf.layers.conv2d_transpose(deconv_4, filters=conv_filters, kernel_size=ksize, strides=(2,2), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_5")
            deconv_5 = tf.layers.batch_normalization(deconv_5, axis=-1, momentum=0.99, epsilon=0.001, name="norm_5")
            deconv_5 = tf.nn.leaky_relu(deconv_5, name="relu_5")
            ##Transpose-Convolution-6 -> (32, 32, 3)
            deconv_6 = tf.layers.conv2d_transpose(deconv_5, filters=self.channels, kernel_size=ksize, strides=(1,1), padding="same", activation=None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_6")
            #Output
            self.output = tf.nn.sigmoid(deconv_6, name="output")

        #Train operations
        with tf.variable_scope("Training"):
            ##Reconstruction loss
            self.loss_reconstruction = tf.reduce_mean(tf.square(tf.subtract(self.x, self.output))) #L2 loss
            ##Global Loss
            self.loss = self.loss_reconstruction
            ##Train ops
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            ##Saver
            self.tf_saver = tf.train.Saver()
            self.train_iteration = start_iteration
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.image("input_images", self.x, max_outputs=8, family="original")
            tf.summary.image("reconstruction_images", self.output, max_outputs=8, family="reconstructed_left")
            tf.summary.scalar("loss", self.loss, family="_loss_main")
            tf.summary.scalar("loss_reconstruction", self.loss_reconstruction, family="losses_reconstruction")
            tf.summary.histogram("hist_style", self.code_style, family="code")

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

    def forward_conditional(self, sess, input_features, input_labels):
        '''Forward step conditioned on the labels
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix of features
        @param input_labels (np.array) array of labels
        @param lambda_e explicit mixing coefficient
        @param lambda_i implicit mixing coefficient
        @return (float) the output  (reconstruction)
        '''
        output = sess.run([self.output], feed_dict={self.x: input_features, self.labels_placeholder: input_labels})
        return output[0]
        
    def test(self, sess, input_features, input_labels):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @param lambda_e explicit mixing coefficient
        @param lambda_i implicit mixing coefficient
        @return (float) the losses: loss, loss_r, loss_c, acc_c, loss_e, loss_i
        '''
        loss, loss_r  = sess.run([self.loss, self.loss_reconstruction], feed_dict={self.x: input_features, self.labels_placeholder: input_labels})
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

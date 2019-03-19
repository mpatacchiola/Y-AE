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

class Autoencoder:
    def __init__(self, batch_size, channels=1, conv_filters=64, style_size=512, content_size=2, ksize=(5,5), start_iteration=0, dir_header="./"):
        '''Init method
        @param sess (tf.Session) the current session
        @param conv_filters_* (int) the number of filters in the convolutional layers
        @param code_size (int) the number of units in the code layer
        @param gradient_clip (bool) applies gradient clipping on the gradient vector
        '''
        #Resource exhausted: OOM when allocating tensor with shape[128,64,32,32]
        self.dir_header = dir_header
        self.start_iteration = start_iteration
        self.channels = channels
        activation_function =  None #tf.nn.leaky_relu
        weight_initializer = None #tf.truncated_normal_initializer(mean=0.0, stddev=0.03) #None
        weight_initializer_implicit = None # tf.random_normal_initializer(mean=0.0, stddev=0.01) #None
        bias_initializer_implicit = None #tf.constant_initializer(-5.0)
        regularizer = None #tf.contrib.layers.l2_regularizer(0.01) #None

        with tf.variable_scope("Input", reuse=False):
            # x is (32 x 32 x 3)
            self.x = tf.placeholder(tf.float32, [batch_size, 32, 32, self.channels]) #Input
            self.labels_placeholder = tf.placeholder(tf.int64,[batch_size])

        ##ROOT
        with tf.variable_scope("Encoder", reuse=False):
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
            ##Conv-7 STYLE
            conv_7 = tf.layers.conv2d(inputs=conv_4, filters=style_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer_implicit, bias_initializer=bias_initializer_implicit, name="conv_7")
            self.code_style = tf.nn.sigmoid(tf.squeeze(conv_7) , name="code_style")
            ##Conv-8 CONTENT
            conv_8 = tf.layers.conv2d(inputs=conv_4, filters=content_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_8")
            self.code_content_logits = tf.squeeze(conv_8) 
            self.code_content = tf.nn.softmax(self.code_content_logits, name="code_content")

        ##LEFT-branch (deteministic content)
        with tf.variable_scope("Decoder", reuse=False):
            self.left_code_content_deterministic = tf.one_hot(indices=self.labels_placeholder, depth=content_size)
            left_code = tf.concat([self.code_style, self.left_code_content_deterministic], axis=1)
            ##Deconvolution-1 -> (2, 2, conv_filters*8)
            left_code_reshaped = tf.reshape(left_code, [batch_size, 1, 1, style_size+content_size])
            left_deconv_1 = tf.layers.conv2d_transpose(left_code_reshaped, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_1")
            left_deconv_1 = tf.layers.batch_normalization(left_deconv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            left_deconv_1 = tf.nn.relu(left_deconv_1, name="relu_1")
            ##Deconvolution-2 -> (4, 4, conv_filters*8)
            left_deconv_2 = tf.layers.conv2d_transpose(left_deconv_1, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_2")
            left_deconv_2 = tf.layers.batch_normalization(left_deconv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            left_deconv_2 = tf.nn.relu(left_deconv_2, name="relu_2")
            ##Deconvolution-3 -> (8, 8, conv_filters*8)
            left_deconv_3 = tf.layers.conv2d_transpose(left_deconv_2, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_3")
            left_deconv_3 = tf.layers.batch_normalization(left_deconv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            left_deconv_3 = tf.nn.relu(left_deconv_3, name="relu_3")
            ##Deconvolution-4 -> (16, 16, conv_filters*8)
            left_deconv_4 = tf.layers.conv2d_transpose(left_deconv_3, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_4")
            left_deconv_4 = tf.layers.batch_normalization(left_deconv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            left_deconv_4 = tf.nn.relu(left_deconv_4, name="relu_4")
            ##Deconvolution-5 -> (32, 32, conv_filters*4)
            left_deconv_5 = tf.layers.conv2d_transpose(left_deconv_4, filters=conv_filters*2, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_5")
            left_deconv_5 = tf.layers.batch_normalization(left_deconv_5, axis=-1, momentum=0.99, epsilon=0.001, name="norm_5")
            left_deconv_5 = tf.nn.relu(left_deconv_5, name="relu_5")
            ##Deconvolution-8 -> (32, 32, 3)
            left_deconv_8 = tf.layers.conv2d_transpose(left_deconv_5, filters=self.channels, kernel_size=ksize, strides=(1,1), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_8")
            #Output
            self.left_output = tf.nn.sigmoid(left_deconv_8, name="output")
            
        with tf.variable_scope("Encoder", reuse=True):
            #Conv-1 -> (16, 16, conv_filters)
            left_conv_1 = tf.layers.conv2d(inputs=self.left_output, filters=conv_filters, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_1")
            left_conv_1 = tf.layers.batch_normalization(left_conv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            left_conv_1 = tf.nn.leaky_relu(left_conv_1, name="relu_1")
            ##Conv-2 -> (8, 8, conv_filters*2)
            left_conv_2 = tf.layers.conv2d(inputs=left_conv_1, filters=conv_filters*2, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_2")
            left_conv_2 = tf.layers.batch_normalization(left_conv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            left_conv_2 = tf.nn.leaky_relu(left_conv_2, name="relu_2")
            ##Conv-3 -> (4, 4, conv_filters*4)
            left_conv_3 = tf.layers.conv2d(inputs=left_conv_2, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_3")
            left_conv_3 = tf.layers.batch_normalization(left_conv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            left_conv_3 = tf.nn.leaky_relu(left_conv_3, name="relu_3")
            ##Conv-4 -> (2, 2, conv_filters*8)
            left_conv_4 = tf.layers.conv2d(inputs=left_conv_3, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_4")
            left_conv_4 = tf.layers.batch_normalization(left_conv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            left_conv_4 = tf.nn.leaky_relu(left_conv_4, name="relu_4")
            ##Conv-7 STYLE
            left_conv_7 = tf.layers.conv2d(inputs=left_conv_4, filters=style_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer_implicit, bias_initializer=bias_initializer_implicit, name="conv_7")
            self.left_code_style = tf.nn.sigmoid(tf.squeeze(left_conv_7) , name="code_style")
            ##Conv-8 CONTENT
            left_conv_8 = tf.layers.conv2d(inputs=left_conv_4, filters=content_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_8")
            self.left_code_content_logits = tf.squeeze(left_conv_8) 
            self.left_code_content = tf.nn.softmax(self.left_code_content_logits, name="code_content")
            self.left_code_content_argmax = tf.argmax(self.left_code_content, axis=1) 

        ##RIGHT-branch (random content)
        ###CODE
        with tf.variable_scope("Decoder", reuse=True):
            self.right_code_content_random = tf.one_hot(indices=tf.random_uniform(shape=[batch_size],minval=0,maxval=content_size,dtype=tf.int32), depth=content_size) #ATTENTION: [minval, maxval) The upperd-bound is EXCLUDED
            #self.right_code_content_random = tf.one_hot(indices=tf.bitwise.invert(self.labels_placeholder), depth=content_size)
            right_code = tf.concat([self.code_style, self.right_code_content_random], axis=1)
            ##Deconvolution-1 -> (2, 2, conv_filters*8)
            right_code_reshaped = tf.reshape(right_code, [batch_size, 1, 1, style_size+content_size])
            right_deconv_1 = tf.layers.conv2d_transpose(right_code_reshaped, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_1")
            right_deconv_1 = tf.layers.batch_normalization(right_deconv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            right_deconv_1 = tf.nn.relu(right_deconv_1, name="relu_1")
            #right_deconv_1 = tf.concat([right_deconv_1, conv_7], axis=3)
            ##Deconvolution-2 -> (4, 4, conv_filters*8)
            right_deconv_2 = tf.layers.conv2d_transpose(right_deconv_1, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_2")
            right_deconv_2 = tf.layers.batch_normalization(right_deconv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            right_deconv_2 = tf.nn.relu(right_deconv_2, name="relu_2")
            #right_deconv_2 = tf.concat([right_deconv_2, conv_6], axis=3)
            ##Deconvolution-3 -> (8, 8, conv_filters*8)
            right_deconv_3 = tf.layers.conv2d_transpose(right_deconv_2, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_3")
            right_deconv_3 = tf.layers.batch_normalization(right_deconv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            right_deconv_3 = tf.nn.relu(right_deconv_3, name="relu_3")
            #right_deconv_3 = tf.concat([right_deconv_3, conv_5], axis=3)
            ##Deconvolution-4 -> (16, 16, conv_filters*8)
            right_deconv_4 = tf.layers.conv2d_transpose(right_deconv_3, filters=conv_filters*4, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_4")
            right_deconv_4 = tf.layers.batch_normalization(right_deconv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            right_deconv_4 = tf.nn.relu(right_deconv_4, name="relu_4")
            #right_deconv_4 = tf.concat([right_deconv_4, conv_4], axis=3)
            ##Deconvolution-5 -> (32, 32, conv_filters*4)
            right_deconv_5 = tf.layers.conv2d_transpose(right_deconv_4, filters=conv_filters*2, kernel_size=ksize, strides=(2,2), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_5")
            right_deconv_5 = tf.layers.batch_normalization(right_deconv_5, axis=-1, momentum=0.99, epsilon=0.001, name="norm_5")
            right_deconv_5 = tf.nn.relu(right_deconv_5, name="relu_5")
            #right_deconv_5 = tf.concat([right_deconv_5, conv_3], axis=3)
            ##Deconvolution-7 -> (128, 128, conv_filters)
            right_deconv_8 = tf.layers.conv2d_transpose(right_deconv_5, filters=self.channels, kernel_size=ksize, strides=(1,1), padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="deconv_8")
            #Output
            self.right_output = tf.nn.sigmoid(right_deconv_8, name="output")
            #mask = tf.random_uniform(dtype=tf.int32, minval=0, maxval=2, shape=tf.shape(right_deconv_8))
            #self.right_output = self.right_output * tf.cast(mask, tf.float32)
                        
        with tf.variable_scope("Encoder", reuse=True):
            #Conv-1 -> (16, 16, conv_filters)
            right_conv_1 = tf.layers.conv2d(inputs=self.right_output, filters=conv_filters, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_1")
            right_conv_1 = tf.layers.batch_normalization(right_conv_1, axis=-1, momentum=0.99, epsilon=0.001, name="norm_1")
            right_conv_1 = tf.nn.leaky_relu(right_conv_1, name="relu_1")
            ##Conv-2 -> (8, 8, conv_filters*2)
            right_conv_2 = tf.layers.conv2d(inputs=right_conv_1, filters=conv_filters*2, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_2")
            right_conv_2 = tf.layers.batch_normalization(right_conv_2, axis=-1, momentum=0.99, epsilon=0.001, name="norm_2")
            right_conv_2 = tf.nn.leaky_relu(right_conv_2, name="relu_2")
            ##Conv-3 -> (4, 4, conv_filters*4)
            right_conv_3 = tf.layers.conv2d(inputs=right_conv_2, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_3")
            right_conv_3 = tf.layers.batch_normalization(right_conv_3, axis=-1, momentum=0.99, epsilon=0.001, name="norm_3")
            right_conv_3 = tf.nn.leaky_relu(right_conv_3, name="relu_3")
            ##Conv-4 -> (2, 2, conv_filters*8)
            right_conv_4 = tf.layers.conv2d(inputs=right_conv_3, filters=conv_filters*4, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_4")
            right_conv_4 = tf.layers.batch_normalization(right_conv_4, axis=-1, momentum=0.99, epsilon=0.001, name="norm_4")
            right_conv_4 = tf.nn.leaky_relu(right_conv_4, name="relu_4")
            ##Conv-7 STYLE
            right_conv_7 = tf.layers.conv2d(inputs=right_conv_4, filters=style_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer_implicit,  bias_initializer=bias_initializer_implicit, name="conv_7")
            self.right_code_style = tf.nn.sigmoid(tf.squeeze(right_conv_7) , name="code_style")
            ##Conv-8 CONTENT
            right_conv_8 = tf.layers.conv2d(inputs=right_conv_4, filters=content_size, strides=(2,2), kernel_size=ksize, padding="same", activation=activation_function, kernel_regularizer=regularizer, kernel_initializer=weight_initializer, name="conv_8")
            self.right_code_content_logits = tf.squeeze(right_conv_8) 
            self.right_code_content = tf.nn.softmax(self.right_code_content_logits, name="code_content")


        #Train operations
        with tf.variable_scope("Training"):

            ##RECONSTRUCTION LOSSES
            #self.loss_reconstruction_left = tf.reduce_mean(tf.abs(tf.subtract(self.x, self.left_output))) #L1 loss (sharper)
            self.loss_reconstruction_left = tf.reduce_mean(tf.square(tf.subtract(self.x, self.left_output))) #L2 loss (sharper)
            

            ##STYLE-CONSISTENCY LOSS
            self.loss_style_consistency = tf.reduce_mean(tf.norm(tf.subtract(self.right_code_style, self.left_code_style)+1.0e-15, ord=2, axis=1)) #adding small value for stability (gradient of sqrt(0) is infinite)

            ##OVERALL Loss
            self.loss = self.loss_reconstruction_left
            
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            #self.train_op = tf.train.RMSPropOptimizer(learning_rate=local_learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(self.loss)
            #self.train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(self.loss)

            self.tf_saver = tf.train.Saver()
            self.train_iteration = start_iteration
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.image("input_images", self.x, max_outputs=15, family="original")

            tf.summary.image("reconstruction_left_images", self.left_output, max_outputs=15, family="reconstructed_left")
            tf.summary.image("reconstruction_right_images", self.right_output, max_outputs=15, family="reconstructed_right")

            tf.summary.scalar("loss", self.loss, family="_loss_main")
            tf.summary.scalar("loss_style_consistency", self.loss_style_consistency, family="losses_style_consistency")
            tf.summary.scalar("loss_reconstruction_left", self.loss_reconstruction_left, family="losses_reconstruction")

            tf.summary.histogram("hist_style", self.code_style, family="code")
            
            #implicit_norm = tf.norm(self.code_style, ord='euclidean', axis=1) #from [b, code] to [b]
            #implicit_mean = tf.reduce_mean(implicit_norm)
            #mean, std = tf.nn.moments(implicit_norm, axes=[0])
            #mean, std = tf.nn.moments(self.code_style, axes=[0,1])
            #tf.summary.scalar("implicit_mean", mean, family="hist_implicit")
            #tf.summary.scalar("implicit_std", std, family="hist_implicit")
            
            mean, std = tf.nn.moments(self.code_style, axes=[0]) #from [b, code] to [code]
            tf.summary.scalar("implicit_batch_std_sum", tf.reduce_sum(std), family="hist_implicit")
            tf.summary.scalar("implicit_batch_mean_sum", tf.reduce_sum(mean), family="hist_implicit")
            mean, std = tf.nn.moments(self.code_style, axes=[1]) #from [b, code] to [b]
            tf.summary.scalar("implicit_sample_std", tf.reduce_mean(std), family="hist_implicit")
            tf.summary.scalar("implicit_sample_mean", tf.reduce_mean(mean), family="hist_implicit")
                                    
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

    def test(self, sess, input_features, input_labels, lambda_e, lambda_i):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        loss, loss_r, loss_i  = sess.run([self.loss, self.loss_reconstruction_left, self.loss_style_consistency], feed_dict={self.x: input_features, self.labels_placeholder: input_labels})
        return loss, loss_r, loss_c, acc_c, loss_e, loss_i

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

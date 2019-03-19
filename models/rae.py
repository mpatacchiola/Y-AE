#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Residual Auto-Encoder
#Based on architecture described in: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

import tensorflow as tf
import numpy as np
import datetime
from time import gmtime, strftime
import os

class Autoencoder:
                
    def __init__(self, batch_size, channels, implicit_size, explicit_size, filters,
                 start_iteration=0, flip=False, dir_header="./"):
        '''Init method
        @param batch_size
        @param channels
        '''
        self.dir_header = dir_header
        self.start_iteration = start_iteration
        activation_function =  tf.nn.leaky_relu
        #weight_initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
        #weight_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        #weight_initializer = tf.initializers.random_uniform(minval=-0.3, maxval=+0.3)       
        weight_initializer = tf.initializers.random_normal(mean=0.0, stddev=0.1)
        #weight_initializer = None
        #with tf.device("/device:GPU:"+str(gpu)):
        ##Input     
        with tf.variable_scope("Input", reuse=False):
            # x is (filters*4, filters*4, 1)
            self.x = tf.placeholder(tf.float32, [batch_size, 128, 128, channels]) #Input
            self.channels = channels
            if(flip): self.x = tf.image.random_flip_left_right(self.x)
            self.labels_placeholder = tf.placeholder(tf.float32,[batch_size, explicit_size])
            self.random_placeholder = tf.placeholder(tf.float32,[batch_size, explicit_size])

        ##Root
        with tf.variable_scope("Encoder", reuse=False):
            #Input: (filters*4, filters*4, 3)        
            bulk_1 = self.conv_bulk(self.x, filters_list=[filters,filters*2,filters*4], strides_list=[(1,1), (2,2), (2,2)], 
                               kernels_list=[(9,9), (3,3), (3,3)], kernel_initializer=weight_initializer, 
                               name="bulk_1")          
            # -> (filters, filters, filters*4)    
            res_2 = self.residual_block(bulk_1, filters=filters*4, kernel_initializer=weight_initializer, name="res_2")   
            res_3 = self.residual_block(res_2, filters=filters*4, kernel_initializer=weight_initializer, name="res_3")   
            res_4 = self.residual_block(res_3, filters=filters*4, kernel_initializer=weight_initializer, name="res_4")      
            # -> (filters, filters, filters*4)   
            bulk_2 = self.conv_bulk(res_4, filters_list=[filters*8,filters*8,filters*8,filters*8], 
                                    strides_list=[(2,2), (2,2), (2,2), (2,2)], 
                                    kernels_list=[(3,3), (3,3), (3,3), (3,3)], kernel_initializer=weight_initializer, 
                                    name="bulk_2")
            # -> (2, 2, filters*8)
            ##Implicit
            conv_implicit = tf.layers.conv2d(inputs=bulk_2, filters=implicit_size, strides=(2,2), kernel_size=(3,3), 
                                             padding="same", activation=None, kernel_initializer=weight_initializer,
                                             name="conv_implicit")
            self.code_implicit = tf.nn.sigmoid(tf.squeeze(conv_implicit) , name="code_implicit")
            ##Explicit
            conv_explicit = tf.layers.conv2d(inputs=bulk_2, filters=explicit_size, strides=(2,2), kernel_size=(3,3),
                                             padding="same", activation=None, kernel_initializer=weight_initializer,
                                             name="conv_explicit")
            self.code_explicit_logits = tf.squeeze(conv_explicit, name="code_explicit")
            self.code_explicit = tf.nn.softmax(self.code_explicit_logits)

        ##LEFT-branch (deteministic content)
        with tf.variable_scope("Decoder", reuse=False):
            left_code = tf.concat([self.code_implicit, self.labels_placeholder], axis=1)
            left_code_reshaped = tf.reshape(left_code, [batch_size, 1, 1, implicit_size+explicit_size])
            # -> (1, 1, imp+exp)
            left_dbulk_1 = self.deconv_bulk(left_code_reshaped, filters_list=[filters*8,filters*8,filters*8,filters*8,filters*4], 
                               strides_list=[(2,2),(2,2),(2,2),(2,2),(2,2)], 
                               kernels_list=[(3,3),(3,3),(3,3),(3,3),(3,3)], 
                               kernel_initializer=weight_initializer, 
                               name="bulk_1")            
            # -> (filters, filters, filters*4)
            left_dres_2 = self.residual_block(left_dbulk_1, filters=filters*4, kernel_initializer=weight_initializer, name="res_2")
            left_dres_3 = self.residual_block(left_dres_2, filters=filters*4, kernel_initializer=weight_initializer, name="res_3") 
            left_dres_4 = self.residual_block(left_dres_3, filters=filters*4, kernel_initializer=weight_initializer, name="res_4")
            left_dres_5 = self.residual_block(left_dres_4, filters=filters*4, kernel_initializer=weight_initializer, name="res_5")
            # -> (filters, filters, filters*4)
            left_dbulk_2 = self.deconv_bulk(left_dres_5, filters_list=[filters*2,filters], 
                                      strides_list=[(2,2),(2,2)], 
                                      kernels_list=[(3,3),(3,3)], 
                                      kernel_initializer=weight_initializer, 
                                      name="bulk_2") 
            # -> (filters*4, filters*4, filters)                               
            conv_output = tf.layers.conv2d(inputs=left_dbulk_2, filters=self.channels, strides=(1,1), 
                                           kernel_size=(3,3), padding="same",
                                           activation=None, kernel_initializer=weight_initializer, name="conv_output")
            self.left_output = tf.nn.sigmoid(conv_output, name="output")                                
            # -> (filters*4, filters*4, 3)                  
        with tf.variable_scope("Encoder", reuse=True):
            #Input: (filters*4, filters*4, 3)        
            left_bulk_1 = self.conv_bulk(self.left_output, filters_list=[filters,filters*2,filters*4],
                                         strides_list=[(1,1), (2,2), (2,2)],
                                         kernels_list=[(9,9), (3,3), (3,3)], kernel_initializer=weight_initializer,
                                         name="bulk_1")
            # -> (filters, filters, filters*4)    
            left_res_2 = self.residual_block(left_bulk_1, filters=filters*4, kernel_initializer=weight_initializer, name="res_2")   
            left_res_3 = self.residual_block(left_res_2, filters=filters*4, kernel_initializer=weight_initializer, name="res_3")   
            left_res_4 = self.residual_block(left_res_3, filters=filters*4, kernel_initializer=weight_initializer, name="res_4")   
            # -> (filters, filters, filters*4)   
            left_bulk_2 = self.conv_bulk(left_res_4, filters_list=[filters*8, filters*8, filters*8, filters*8], 
                                         strides_list=[(2,2), (2,2), (2,2), (2,2)], 
                                         kernels_list=[(3,3), (3,3), (3,3), (3,3)], kernel_initializer=weight_initializer, 
                                         name="bulk_2")
            # -> (2, 2, filters*8)
            ##Implicit
            left_conv_implicit = tf.layers.conv2d(inputs=left_bulk_2, filters=implicit_size, strides=(2,2), kernel_size=(3,3), 
                                                  padding="same", activation=None, kernel_initializer=weight_initializer,
                                                  name="conv_implicit")
            self.left_code_implicit = tf.nn.sigmoid(tf.squeeze(left_conv_implicit) , name="code_implicit")
            ##Explicit
            left_conv_explicit = tf.layers.conv2d(inputs=left_bulk_2, filters=explicit_size, strides=(2,2), kernel_size=(3,3),
                                                  padding="same", activation=None, kernel_initializer=weight_initializer,
                                                  name="conv_explicit")
            self.left_code_explicit_logits = tf.squeeze(left_conv_explicit, name="code_explicit")
            self.left_code_explicit = tf.nn.softmax(self.left_code_explicit_logits)                               

        ##RIGHT-branch (random content)
        with tf.variable_scope("Decoder", reuse=True):
            right_code = tf.concat([self.code_implicit, self.random_placeholder], axis=1)
            right_code_reshaped = tf.reshape(right_code, [batch_size, 1, 1, implicit_size+explicit_size])
            # -> (1, 1, imp+exp)
            right_dbulk_1 = self.deconv_bulk(right_code_reshaped, filters_list=[filters*8,filters*8,filters*8,filters*8,filters*4], 
                                     strides_list=[(2,2),(2,2),(2,2),(2,2),(2,2)], 
                                     kernels_list=[(3,3),(3,3),(3,3),(3,3),(3,3)], 
                                     kernel_initializer=weight_initializer, 
                                     name="bulk_1")
            # -> (filters, filters, filters*4)
            right_dres_2 = self.residual_block(right_dbulk_1, filters=filters*4,kernel_initializer=weight_initializer, name="res_2")
            right_dres_3 = self.residual_block(right_dres_2, filters=filters*4, kernel_initializer=weight_initializer, name="res_3")
            right_dres_4 = self.residual_block(right_dres_3, filters=filters*4, kernel_initializer=weight_initializer, name="res_4")
            right_dres_5 = self.residual_block(right_dres_4, filters=filters*4, kernel_initializer=weight_initializer, name="res_5")
            # -> (filters, filters, filters*4)
            right_dbulk_2 = self.deconv_bulk(right_dres_5, filters_list=[filters*2,filters], 
                               strides_list=[(2,2),(2,2)], 
                               kernels_list=[(3,3),(3,3)], 
                               kernel_initializer=weight_initializer, 
                               name="bulk_2") 
            # -> (filters*4, filters*4, filters)                               
            right_conv_output = tf.layers.conv2d(inputs=right_dbulk_2, filters=self.channels, strides=(1,1), 
                                                 kernel_size=(3,3), padding="same",
                                                 activation=None, kernel_initializer=weight_initializer, name="conv_output")
            self.right_output = tf.nn.sigmoid(right_conv_output, name="output")                                
            # -> (filters*4, filters*4, 3)  
        with tf.variable_scope("Encoder", reuse=True):                       
            #Input: (filters*4, filters*4, 3)    
            right_bulk_1 = self.conv_bulk(self.right_output, filters_list=[filters,filters*2,filters*4], strides_list=[(1,1), (2,2), (2,2)], 
                                   kernels_list=[(9,9), (3,3), (3,3)], kernel_initializer=weight_initializer, 
                                   name="bulk_1")          
            # -> (filters, filters, filters*4)    
            right_res_2 = self.residual_block(right_bulk_1, filters=filters*4, kernel_initializer=weight_initializer, name="res_2")
            right_res_3 = self.residual_block(right_res_2, filters=filters*4, kernel_initializer=weight_initializer, name="res_3")
            right_res_4 = self.residual_block(right_res_3, filters=filters*4, kernel_initializer=weight_initializer, name="res_4")
            # -> (filters, filters, filters*4)   
            right_bulk_2 = self.conv_bulk(right_res_4, filters_list=[filters*8, filters*8, filters*8, filters*8],
                                          strides_list=[(2,2), (2,2), (2,2), (2,2)], 
                                          kernels_list=[(3,3), (3,3), (3,3), (3,3)], kernel_initializer=weight_initializer, 
                                          name="bulk_2")
            # -> (2, 2, filters*8)
            ##Implicit
            right_conv_implicit = tf.layers.conv2d(inputs=right_bulk_2, filters=implicit_size, strides=(2,2), kernel_size=(3,3), 
                                                   padding="same", activation=None, kernel_initializer=weight_initializer,
                                                   name="conv_implicit")
            self.right_code_implicit = tf.nn.sigmoid(tf.squeeze(right_conv_implicit) , name="code_implicit")
            ##Explicit
            right_conv_explicit = tf.layers.conv2d(inputs=right_bulk_2, filters=explicit_size, strides=(2,2), kernel_size=(3,3),
                                                   padding="same", activation=None, kernel_initializer=weight_initializer,
                                                   name="conv_explicit")
            self.right_code_explicit_logits = tf.squeeze(right_conv_explicit, name="code_explicit")
            self.right_code_explicit = tf.nn.softmax(self.right_code_explicit_logits)   


        #Train operations
        with tf.variable_scope("Training"):
            ##Losses
            #self.loss_r = tf.reduce_mean(tf.abs(tf.subtract(self.x, self.left_output))) #L1 loss
            self.loss_r = tf.reduce_mean(tf.pow(tf.subtract(self.x, self.left_output), 2)) #L2 loss
            #self.loss_r = tf.reduce_mean(tf.abs(tf.pow(tf.subtract(self.x, self.left_output), 3))) #L3 loss
                       
            self.loss_c = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_placeholder, 
                                                          logits=self.code_explicit_logits)
            self.loss_e = tf.losses.softmax_cross_entropy(onehot_labels=self.random_placeholder, 
                                                          logits=self.right_code_explicit_logits)
            self.loss_i = tf.reduce_mean(tf.norm(
                                         tf.subtract(self.right_code_implicit, self.left_code_implicit)
                                         +1.0e-15, ord=2, axis=1)) #small value for stability (gradient(sqrt(0)) is infinite)
            lambda_i = 0.01
            lambda_c = 0.1
            self.loss = self.loss_r + tf.multiply(lambda_c,self.loss_c) + self.loss_e + tf.multiply(lambda_i,self.loss_i)
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                   beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            #self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.loss)
            #self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, 
            #                                          decay=0.9, momentum=0.9, epsilon=1e-10).minimize(self.loss)

            self.tf_saver = tf.train.Saver()
            self.train_iteration = start_iteration
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.image("input_images", self.x, max_outputs=4, family="original")

            tf.summary.image("reconstruction_left_images", self.left_output, max_outputs=4, family="reconstructed")
            tf.summary.image("reconstruction_right_images", self.right_output, max_outputs=4, family="reconstructed")

            tf.summary.scalar("loss", self.loss, family="1_loss_main")
            tf.summary.scalar("loss_r", self.loss_r, family="2_loss_reconstruction")
            tf.summary.scalar("loss_c", self.loss_c, family="3_loss_supervised")
            tf.summary.scalar("loss_i", self.loss_i, family="4_loss_implicit_consistency")
            tf.summary.scalar("loss_e", self.loss_e, family="5_loss_explicit")

            tf.summary.histogram("hist_implicit", self.code_implicit, family="code")

    def init_summary(self, sess):
        summary_folder = self.dir_header + '/log/' + str(datetime.datetime.now().time()) + '_iter_' + str(self.start_iteration)
        self.tf_summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
        self.summaries = tf.summary.merge_all() #merge all the previous summaries

    def conv_bulk(self, x, filters_list, strides_list, kernels_list, kernel_initializer, name):
        tot_elements = len(filters_list)
        for i in range(tot_elements):           
            conv = tf.layers.conv2d(inputs=x, filters=filters_list[i], strides=strides_list[i], 
                                    kernel_size=kernels_list[i], padding="same",
                                    activation=None, kernel_initializer=kernel_initializer, name=name+"_conv_"+str(i))
            conv = tf.layers.batch_normalization(conv, axis=-1, momentum=0.99, epsilon=0.001, name=name+"_norm_"+str(i))
            conv = tf.nn.leaky_relu(conv, name=name+"_relu_"+str(i))
            x = conv        
        return conv

    def residual_block(self, x, filters, kernel_initializer, name):
        conv_1 = tf.layers.conv2d(inputs=x, filters=filters, strides=(1,1), kernel_size=(3,3), padding="same",
                                      activation=None, kernel_initializer=kernel_initializer, name=name+"_conv_1")
        conv_1 = tf.layers.batch_normalization(conv_1, axis=-1, momentum=0.99, epsilon=0.001, name=name+"_norm_1")
        conv_1 = tf.nn.leaky_relu(conv_1, name=name+"_relu_1")  
        conv_2 = tf.layers.conv2d(inputs=conv_1, filters=filters, strides=(1,1), kernel_size=(3,3), padding="same",
                                  activation=None, kernel_initializer=kernel_initializer, name=name+"_conv_2")
        conv_2 = tf.layers.batch_normalization(conv_2, axis=-1, momentum=0.99, epsilon=0.001, name=name+"_norm_2")        
        shortcut = tf.add(x, conv_2, name=name+"_residual") #residual
        y = tf.nn.leaky_relu(shortcut, name=name+"_output")
        return y

    def deconv_bulk(self, x, filters_list, strides_list, kernels_list, kernel_initializer, name):
        tot_elements = len(filters_list)
        for i in range(tot_elements):                                   
            deconv = tf.layers.conv2d_transpose(x, filters=filters_list[i], strides=strides_list[i], padding="same",
                                                kernel_size=kernels_list[i],kernel_initializer=kernel_initializer,
                                                activation=None, name=name+"_deconv_"+str(i))             
            deconv = tf.layers.batch_normalization(deconv, axis=-1, momentum=0.99, epsilon=0.001, name=name+"_norm_"+str(i))
            deconv = tf.nn.leaky_relu(deconv, name=name+"_relu_"+str(i))
            x = deconv        
        return deconv
        
    def forward(self, sess, input_feature):
        '''Feed-forward pass in the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (np.array) the output of the autoencoder
        '''
        output = sess.run([self.output], feed_dict={self.x: input_feature})
        return output

    def test(self, sess, input_feature, input_labels):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return the output image
        '''
        out = sess.run([self.left_output], feed_dict={self.x: input_feature, self.labels_placeholder: input_labels})
        return out

    def train(self, sess, input_features, input_labels, random_labels, learning_rate, iteration, summary_rate):
        '''Single step training of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        _, loss, summ = sess.run([self.train_op, self.loss, self.summaries], feed_dict={self.x: input_features, self.labels_placeholder: input_labels, self.random_placeholder: random_labels, self.learning_rate: learning_rate})
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
        time_string = strftime("%d%m%Y_%H%M%S", gmtime())
        model_folder = self.dir_header + "/model/" + str(time_string) + "_" + str(self.train_iteration) + "/model.ckpt"
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


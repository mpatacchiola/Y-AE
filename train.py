#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
import datetime
from time import gmtime, strftime
import argparse
from dataset import Dataset

def return_onehot_random_labels(batch_size, tot_values):
    random_labels = np.zeros([batch_size, tot_values], dtype=np.float32)
    arange = np.arange(batch_size, dtype=np.int32)
    rchoice = np.random.choice(tot_values, size=batch_size, replace=True)
    random_labels[arange, rchoice] = 1.0
    return random_labels
    
def main():
    ##Defining the parser
    parser = argparse.ArgumentParser(description="Tensorflow Trainer")
    parser.add_argument("--resume", type=str, help="resume from checkpoint: ./path/model.ckpt")
    parser.add_argument("--start_iteration", default=0, type=int, help="starting iterations")
    parser.add_argument("--stop_iteration", default=1000, type=int, help="starting iterations")
    parser.add_argument("--dataset", default="mnist", type=str, help="dataset to use for training")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index")
    parser.add_argument("--arch", default="yae", type=str, help="architecture to use for training: yae, cae")
    parser.add_argument("--lambda_e", default=1.0, type=float, help="Explicit loss mixing coefficient")
    parser.add_argument("--lambda_i", default=1.0, type=float, help="Implicit loss mixing coefficient")
    args = parser.parse_args()
   
    #Set the GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)    
    import tensorflow as tf
 
    ##Set the hyper-parameters based on the chosen dataset
    if(args.dataset=="mnist" and (args.arch=="yae" or args.arch=="cae")):
        learning_rate = 0.0001
        mini_batch_size = 128
        tot_labels = 10
        tot_epochs = 100
        dataset_size = 60000
        tot_iterations = int((dataset_size / mini_batch_size) * tot_epochs)
        save_every_iteration = tot_iterations-1
        print_every_iteration = 25
        chunk_index = None
        tot_chunks = None
        simulation_path = "./experiments/mnist"
        features_path = "./datasets/mnist/train/features.npy"
        labels_path = "./datasets/mnist/train/labels.npy"
        dataset_train = Dataset()
        dataset_train.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=True, verbose=True)
        features_path = "./datasets/mnist/test/features.npy"
        labels_path = "./datasets/mnist/test/labels.npy"
        dataset_test = Dataset()
        dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=False, verbose=True)
    elif(args.dataset=="mnist" and args.arch=="lenet"):
        learning_rate = 0.0001
        mini_batch_size = 128
        tot_labels = 10
        tot_epochs = 30
        dataset_size = 60000
        tot_iterations = int((dataset_size / mini_batch_size) * tot_epochs)
        save_every_iteration = tot_iterations-1
        print_every_iteration = 25
        chunk_index = None
        tot_chunks = None
        simulation_path = "./experiments/mnist"
        features_path = "./datasets/mnist/train/features.npy"
        labels_path = "./datasets/mnist/train/labels.npy"
        dataset_train = Dataset()
        dataset_train.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=True, verbose=True)
        features_path = "./datasets/mnist/test/features.npy"
        labels_path = "./datasets/mnist/test/labels.npy"
        dataset_test = Dataset()
        dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=False, verbose=True)    
    elif(args.dataset=="svhn"):
        learning_rate = 0.0001
        mini_batch_size = 128
        chunk_index = None
        tot_chunks = None
    elif(args.dataset=="chairs"):
        learning_rate = 0.0000125 #0.00005 # [0.0000025 #0.000005 # 0.00001]
        mini_batch_size = 64
        tot_labels = 31
        tot_epochs = 1000
        save_every_iteration = 3000
        print_every_iteration = 25
        reshuffle_every_iteration = 5000
        chunk_index = 0
        tot_chunks = 6
        dataset_size = 80166 #86366 - (100*62) -> (tot_types*62) - (test_types*62)
        chunk_size = dataset_size / tot_chunks
        chunk_every_iteration = int(chunk_size / mini_batch_size)
        tot_iterations = int((dataset_size / mini_batch_size) * tot_epochs)            
        dataset_train = Dataset()
        features_path = "./datasets/chairs/train/features.npy"
        labels_path = "./datasets/chairs/train/labels.npy"
        training_path = "./datasets/chairs/train"
        simulation_path = "./experiments/chairs"
        #dataset_train.split_chunks(features_path,labels_path,
        #                           output_path=training_path,
        #                           tot_chunks=tot_chunks, shuffle=True)
        dataset_train.load_chunk(chunk_index, tot_labels, root=training_path, normalizer=255.0)
    else:
        raise ValueError("[ERROR] The dataset '" + args.dataset + "' does not exist!")     


    if(args.arch=="yae" and args.dataset=="mnist"):
            from models.yae_mnist_svhn import Autoencoder
            my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=32, content_size=10, 
                                         ksize=(3,3), start_iteration=args.start_iteration, dir_header=simulation_path)
    elif(args.arch=="yae" and args.dataset=="chairs"):
            from models.yae_chairs import Autoencoder
            my_net = Autoencoder(batch_size=mini_batch_size, conv_filters=64, style_size=481, content_size=31,
                                         ksize=(3,3), start_iteration=args.start_iteration, dir_header=simulation_path)
    elif(args.arch=="cae" and args.dataset=="mnist"):
            from models.cae_mnist_svhn import Autoencoder
            my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=32, content_size=10, 
                                         ksize=(3,3), start_iteration=args.start_iteration, dir_header=simulation_path)
    elif(args.arch=="lenet" and args.dataset=="mnist"):
            from models.lenet_mnist_svhn import LeNet
            my_net = LeNet(batch_size=mini_batch_size, tot_labels=10, channels=1, conv_filters=8, 
                                         ksize=(3,3), start_iteration=args.start_iteration, dir_header=simulation_path)
    else:
        raise ValueError("[ERROR] The architecture '" + args.arch + "' does not exist!") 
             
    #Init the session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    my_net.init_summary(sess)
                          
    if args.resume:
        print("[INFO] Resuming from checkpoint: " + str(args.resume))
        my_net.load(sess, args.resume)
    else:                           
        sess.run(tf.global_variables_initializer()) #WARNING: do not call it when the load() method is used

    print("[INFO] Starting training...")
    for iteration in range(args.start_iteration, tot_iterations):
        
        if(args.dataset=="chairs"):
            input_features, input_labels = dataset_train.return_features_labels(mini_batch_size, onehot=True)
            random_labels = return_onehot_random_labels(batch_size=mini_batch_size, tot_values=tot_labels)
            local_loss = my_net.train(sess, input_features, input_labels, random_labels, 
                                              learning_rate, args.lambda_e, args.lambda_i, iteration, print_every_iteration)
        elif(args.arch=="yae" and args.dataset=="mnist"):
            input_features, input_labels = dataset_train.return_features_labels(mini_batch_size, onehot=False)
            local_loss = my_net.train(sess, input_features, input_labels, 
                                              learning_rate, args.lambda_e, args.lambda_i, iteration, print_every_iteration)
        elif(args.arch=="cae" and args.dataset=="mnist"):
            input_features, input_labels = dataset_train.return_features_labels(mini_batch_size, onehot=False)
            local_loss = my_net.train(sess, input_features, input_labels, 
                                              learning_rate, iteration, print_every_iteration)
        elif(args.arch=="lenet" and args.dataset=="mnist"):
            input_features, input_labels = dataset_train.return_features_labels(mini_batch_size, onehot=True)
            local_loss = my_net.train(sess, input_features, input_labels, 
                                      learning_rate, iteration, print_every_iteration)        
        else:
            raise ValueError("[ERROR] The combination of this dataset and architecture is not supported!") 
                
        if(iteration % print_every_iteration == 0):
            print("Iteration: " + str(iteration) + "/" + str(tot_iterations) + " [" + str(round((iteration/float(tot_iterations))*100.0, 1)) + "%]")
            print("Loss: " + str(local_loss))
            #print("Labels: " + str(np.argmax(input_labels, axis=1)))
            #print("Labels (random): " + str(np.argmax(random_labels, axis=1)))
            print("Features value max: " + str(np.amax(input_features)))
            print("Learning rate: " + str(learning_rate))
            print("")
            
            #print('Test: [{0}][{1}]\t'
            #      'Loss {loss:.4f}\t'.format(
            #       iteration, tot_iterations, loss=local_loss))
            
            #TODO remove
            #for i in range(mini_batch_size):
            #    import cv2
            #    img = input_features[i, :, :, :] * 255
            #    label = input_labels[i,:]
            #    cv2.imwrite('./samples/' + str(i) + '_' + str(np.argmax(label)) + '.jpg', img)
            #return 
        if(iteration % save_every_iteration == 0 and iteration!=0):
                my_net.save(sess, verbose=True)
        if(chunk_index is not None):
            if(iteration % chunk_every_iteration == 0 and iteration!=0 and tot_chunks>1):
                chunk_index+=1
                if(chunk_index >= tot_chunks): chunk_index=0
                dataset_train.load_chunk(chunk_index, tot_labels, root=training_path, normalizer=255.0)
            if(iteration % reshuffle_every_iteration == 0 and iteration!=0):
                dataset_train.split_chunks(features_path, labels_path, training_path, tot_chunks=tot_chunks, shuffle=True)
                dataset_train.load_chunk(chunk_index, tot_labels, root=training_path, normalizer=255.0)


if __name__ == "__main__":
    main()

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
from io import BytesIO
import argparse
from dataset import Dataset
import cv2

def test_chairs(sess, my_autoencoder):
    input_labels = np.identity(31)

    print("Starting training...")
    random_indices = np.random.randint(0, test_features_dataset.shape[0], size=batch_size)
    for i in random_indices:
        input_features = test_features_dataset[i,:,:,:]
        input_features_list = [input_features] * 31
        input_features_array = np.stack(input_features_list, 0) 
        print(input_features_array.shape)
        output = my_autoencoder.test(sess, input_features_array, input_labels)
        output = output[0]
        print(output.shape)
        cv2.imwrite("./samples/" + str(i) + ".jpg", (input_features*255).astype(np.uint8))
        for j in range(31):           
            cv2.imwrite("./samples/" + str(i) + "_" + str(j) + ".jpg", (output[j,:,:,:]*255).astype(np.uint8))



def main():
    ##Defining the parser
    parser = argparse.ArgumentParser(description="Tensorflow Trainer")
    parser.add_argument("--resume", type=str, help="resume from checkpoint: ./path/model.ckpt")
    parser.add_argument("--iter", default=0, type=int, help="starting iterations")
    parser.add_argument("--dataset", default="mnist", type=str, help="dataset to use for training")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index")
    parser.add_argument("--lambda_e", default=1.0, type=float, help="Explicit loss mixing coefficient")
    parser.add_argument("--lambda_i", default=1.0, type=float, help="Implicit loss mixing coefficient")
    args = parser.parse_args()
    start_iteration = args.iter
    GPU = args.gpu

    #Set the GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)    
    import tensorflow as tf
    from tensorflow.python.lib.io import file_io
         
    ##Set the hyper-parameters based on the chosen dataset
    if(args.dataset=="mnist"):
        tot_labels = 10
        learning_rate = 0.0001
        mini_batch_size = 10000
        chunk_index = None
        tot_chunks = None
        dataset_test = Dataset()
        features_path = "./datasets/mnist/test/features.npy"
        labels_path = "./datasets/mnist/test/labels.npy"
        simulation_path = "./experiments/mnist"
        dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=False, verbose=True)
        from models.yae_mnist_svhn import Autoencoder
        my_autoencoder = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=32, content_size=10, 
                                     ksize=(3,3), start_iteration=0, dir_header=simulation_path)
    elif(args.dataset=="svhn"):
        learning_rate = 0.0001
        mini_batch_size = 128
        chunk_index = None
        tot_chunks = None
    elif(args.dataset=="chairs"):
        tot_labels = 31
        mini_batch_size = 8
        dataset_size = 80166 #86366 - (100*62) -> (tot_types*62) - (test_types*62)          
        dataset_test = Dataset()
        features_path = "./datasets/chairs/test/features.npy"
        labels_path = "./datasets/chairs/test/labels.npy"
        test_path = "./simulations/chairs/test"
        simulation_path = "./simulations/chairs"
        #dataset_train.split_chunks(features_path,labels_path,
        #                           output_path=training_path,
        #                           tot_chunks=tot_chunks, shuffle=True)
        dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=False, verbose=True)
        from models.rae import Autoencoder
        #The input batch size is given here by the number of tot_labels
        my_autoencoder = Autoencoder(batch_size=31, channels=1, implicit_size=481, explicit_size=31, filters=8, 
                                     start_iteration=start_iteration, flip=False, dir_header=simulation_path)
    else:
        raise ValueError("[ERROR] The dataset does not exist!")     
  
    #Init the session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    my_autoencoder.init_summary(sess)
                          
    if args.resume:
        print("[INFO] Resuming from checkpoint: " + str(args.resume))
        my_autoencoder.load(sess, args.resume)
    else:                           
        raise ValueError("[ERROR] To test a model it is necessary to resume from checkpoint...")

    if(args.dataset=="chairs"):
        print("Starting chairs test...")
        input_features, _ = dataset_test.return_features_labels(mini_batch_size, onehot=True)
        input_labels = np.identity(31)
        for i in range(mini_batch_size):
            input_feature_single = input_features[i,:,:,:]
            input_features_list = [input_feature_single] * 31
            input_features_array = np.stack(input_features_list, 0) 
            print(input_features_array.shape)
            output = my_autoencoder.test(sess, input_features_array, input_labels)
            output = output[0]
            print(output.shape)
            cv2.imwrite(test_path + "/" + str(i) + ".jpg", (input_feature_single*255).astype(np.uint8))
            for j in range(31):           
                cv2.imwrite(test_path + "/" + str(i) + "_" + str(j) + ".jpg", (output[j,:,:,:]*255).astype(np.uint8))
        print("Done!")
        
    elif(args.dataset=="mnist"):
        input_features, input_labels = dataset_test.return_features_labels(mini_batch_size, onehot=False)
        loss, loss_r, loss_c, acc_c, loss_e, loss_i = my_autoencoder.test(sess, input_features, input_labels,
                                                                          args.lambda_e, args.lambda_i)
        with open(simulation_path + "/test_mnist.csv", "w") as text_file:
            header = "loss, loss_r, loss_c, acc_c, loss_e, loss_i"
            text_file.write(header + '\n' + str(loss)+","+ str(loss_r)+","+str(loss_c)+","+str(acc_c)+","+str(loss_e)+","+ str(loss_i))
            
    
if __name__ == "__main__":
    main()

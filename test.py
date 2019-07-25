#The MIT License (MIT)
#Copyright (c) 2019 anonymous authors
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
from time import gmtime, strftime
import argparse
from dataset import Dataset
import cv2
import scipy.signal
from skimage.measure import compare_ssim

def main():
    ##Defining the parser
    parser = argparse.ArgumentParser(description="Tensorflow Trainer")
    parser.add_argument("--resume", type=str, help="the path to the saved network")
    parser.add_argument("--path", type=str, help="the root folder of the experiment (it contains the logs and model)")
    parser.add_argument("--gendata_path", type=str, help="the folder containing the artificial data")    
    parser.add_argument("--load", type=str, help="load data")
    parser.add_argument("--type", default="gendata", type=str, help="type of test: gendata (generate a dataset), loss (return losses)")
    parser.add_argument("--arch", default="yae", type=str, help="architecture to use for training: yae, cae")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index")
    parser.add_argument("--lambda_e", default=1.0, type=float, help="Explicit loss mixing coefficient")
    parser.add_argument("--lambda_i", default=1.0, type=float, help="Implicit loss mixing coefficient")
    parser.add_argument("--implicit_units", default=32, type=int, help="implicit units in the code")
    parser.add_argument("--tot_samples", default=16, type=int, help="the total sample generted by measure")
    parser.add_argument("--batch", default=20000, type=int, help="the batch feed to the LeNet classifier")
    args = parser.parse_args()
    
    #Set the GPU
    GPU = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)    
    import tensorflow as tf

    tot_labels = 10
    mini_batch_size = 10000
    dataset_test = Dataset()
    features_path = "./datasets/mnist/test/features.npy"
    labels_path = "./datasets/mnist/test/labels.npy"
    dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=False, verbose=True)   
          
    ##Set the hyper-parameters based on the chosen dataset
    if(args.arch=="yae"):
        simulation_path = args.path
        from models.yae import Autoencoder
        my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=args.implicit_units, content_size=10, ksize=(3,3), start_iteration=0, dir_header=simulation_path)
    elif(args.arch=="cae"):
        simulation_path = args.path
        from models.cae import Autoencoder
        my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=args.implicit_units, content_size=10, ksize=(3,3), start_iteration=0, dir_header=simulation_path)
    elif(args.arch=="cvae"):
        simulation_path = args.path
        from models.cvae import Autoencoder
        my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=8, style_size=args.implicit_units, content_size=10, ksize=(3,3), start_iteration=0, dir_header=simulation_path)
    elif(args.arch=="aae"):
        simulation_path = args.path
        from models.aae import Autoencoder
        my_net = Autoencoder(batch_size=mini_batch_size, channels=1, conv_filters=4, style_size=args.implicit_units, content_size=10, ksize=(3,3), start_iteration=0, dir_header=simulation_path)
    elif(args.arch=="lenet"):
        simulation_path = args.path
        from models.lenet import LeNet
        my_net = LeNet(batch_size=args.batch, channels=1, conv_filters=8, tot_labels=10, ksize=(5,5), start_iteration=0, dir_header=simulation_path)                
    else:
        raise ValueError("[ERROR] The dataset does not exist!")     
  
    #Init the session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                          
    if args.resume:
        print("[INFO] Resuming from checkpoint: " + str(args.resume))
        my_net.load(sess, args.resume)
    else:                           
        raise ValueError("[ERROR] To test a model it is necessary to resume from checkpoint...")

    if(args.arch=="yae" and args.type=="loss"):
        if not os.path.exists(simulation_path + "/test_loss"): os.makedirs(simulation_path + "/test_loss")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        test_loss_path = simulation_path + "/test_loss/" + time_id
        os.makedirs(test_loss_path)
        input_features, input_labels = dataset_test.return_features_labels(mini_batch_size, onehot=False)
        loss, loss_r, loss_c, acc_c, loss_e, loss_i = my_net.test(sess, input_features, input_labels,
                                                                  args.lambda_e, args.lambda_i)
        with open(test_loss_path + "/test_loss.csv", "w") as text_file:
            header = "loss, loss_r, loss_c, acc_c, loss_e, loss_i"
            body = str(loss)+","+ str(loss_r)+","+str(loss_c)+","+str(acc_c)+","+str(loss_e)+","+ str(loss_i)
            text_file.write(header + '\n' + body)
            print("====================================")
            print(header)
            print(body)
            print("====================================")
    elif(args.arch=="cae" and args.type=="loss"):
        if not os.path.exists(simulation_path + "/test_loss"): os.makedirs(simulation_path + "/test_loss")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        test_loss_path = simulation_path + "/test_loss/" + time_id
        os.makedirs(test_loss_path)
        input_features, input_labels = dataset_test.return_features_labels(mini_batch_size, onehot=False)
        loss = my_net.test(sess, input_features, input_labels)
        with open(test_loss_path + "/test_loss.csv", "w") as text_file:
            header = "loss"
            body = str(loss)
            text_file.write(header + '\n' + body)
            print("====================================")
            print(header)
            print(body)
            print("====================================")
    elif(args.arch=="cvae" and args.type=="loss"):
        if not os.path.exists(simulation_path + "/test_loss"): os.makedirs(simulation_path + "/test_loss")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        test_loss_path = simulation_path + "/test_loss/" + time_id
        os.makedirs(test_loss_path)
        input_features, input_labels = dataset_test.return_features_labels(mini_batch_size, onehot=False)
        loss = my_net.test(sess, input_features, input_labels)
        with open(test_loss_path + "/test_loss.csv", "w") as text_file:
            header = "loss_r"
            body = str(loss)
            text_file.write(header + '\n' + body)
            print("====================================")
            print(header)
            print(body)
            print("====================================")
    elif((args.arch=="cae" or args.arch=="cvae" or args.arch=="yae" or args.arch=="aae") and args.type=="gendata"):
        if not os.path.exists(simulation_path + "/gendata"): os.makedirs(simulation_path + "/gendata")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        gendata_path = simulation_path + "/gendata/" + time_id
        os.makedirs(gendata_path)
        print("Starting gendata...")
        input_features, _ = dataset_test.return_features_labels(10000, onehot=False, shuffle=False)
        features_list = list()
        labels_list = list()
        for i in range(0, 10):
            print("Input: " + str(i))
            print("Input (shape): " + str(input_features.shape))
            input_labels = np.ones(10000) * i
            if(args.arch=="cae" or args.arch=="cvae" or args.arch=="aae"): output = my_net.forward_conditional(sess, input_features, input_labels) #size [10000, 32, 32, 1]
            elif(args.arch=="yae"): output = my_net.forward_conditional(sess, input_features, input_labels, args.lambda_e, args.lambda_i)
            print("Output (shape): " + str(output.shape))
            output = (output * 255).astype(np.uint8)
            features_list.append(output)
            labels_list.append(input_labels)
        print("Saving data...")
        features_matrix = np.concatenate(features_list, axis=0)
        print("Features (shape): " + str(features_matrix.shape))
        np.save(gendata_path + "/features", features_matrix)
        labels_matrix = np.concatenate(labels_list, axis=0)
        print("Labels (shape): " + str(labels_matrix.shape))
        np.save(gendata_path + "/labels", labels_matrix)
        print("Done!")
        
    elif(args.type=="metrics"):
        print("Iterating the test set...")
        original_data_matrix = np.load(features_path)
        data_matrix = np.load(args.gendata_path + "/features.npy")
        if not os.path.exists(simulation_path + "/sample"): os.makedirs(simulation_path + "/sample")
        if not os.path.exists(simulation_path + "/metrics"): os.makedirs(simulation_path + "/metrics")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        sample_path = simulation_path + "/sample/" + time_id
        metrics_path = simulation_path + "/metrics/" + time_id
        os.makedirs(sample_path)
        os.makedirs(metrics_path)
        img_ssim_list = list()
        img_mse_list = list()
        for i in range(10000):
            img_original = original_data_matrix[i, :, :, 0]
            if(i<args.tot_samples): cv2.imwrite(sample_path + "/" + str(i) + ".png", img_original)
            for j in range(0, 10):
                location = (j*10000)+i
                img_generated = data_matrix[location, :, :, 0]
                (score, diff) = compare_ssim(img_original, img_generated, full=True)
                mse = (img_original-img_generated)**2
                img_ssim_list.append(score)
                img_mse_list.append(mse)
                if(i<args.tot_samples): cv2.imwrite(sample_path + "/" + str(i) + "_" + str(j) + ".png", img_generated)
            if(i%1000==0):
                print(str(i) + "/" + str(10000))
        print("====================================")        
        print("INTERNAL SSIM")
        print("Mean: \t" + str(np.mean(img_ssim_list)))
        print("Std: \t" + str(np.std(img_ssim_list)))
        print("------------------------------------")
        print("INTERNAL MSE")
        print("Mean: \t" + str(np.mean(img_mse_list)))
        print("Std: \t" + str(np.std(img_mse_list)))
        print("====================================")
        with open(metrics_path + "/test_metrics.csv", "w") as text_file:
            header = "SSIM, MSE"
            body = str(np.mean(img_ssim_list)) + ',' + str(np.mean(img_mse_list))
            text_file.write(header + '\n' + body)
            
    elif(args.arch=="lenet" and args.type=="accuracy"):
        if not os.path.exists(simulation_path + "/test_gendata_accuracy"): os.makedirs(simulation_path + "/test_gendata_accuracy")
        time_id = strftime("%H%M%S_%d%m%Y", gmtime())
        accuracy_path = simulation_path + "/test_gendata_accuracy/" + time_id
        os.makedirs(accuracy_path)
        dataset_test = Dataset()
        features_path = args.gendata_path + "/features.npy"
        labels_path = args.gendata_path + "/labels.npy"
        dataset_test.load(features_path, labels_path, tot_labels, normalizer=255.0, shuffle=True, verbose=True) 
        input_features, input_labels = dataset_test.return_features_labels(args.batch, onehot=False)
        print(input_features.shape)
        output = my_net.test(sess, input_features, input_labels)
        print("==============================")
        print("Loss:\t\t" + str(output[0]))
        print("Accuracy:\t" + str(output[1] * 100.0))
        print("==============================")
        with open(accuracy_path + "/test_gendata_accuracy.csv", "w") as text_file:
            header = "Loss, Accuracy"
            body = str(output[0]) + ',' + str(output[1] * 100.0)
            text_file.write(header + '\n' + body)

    else:
        raise ValueError("[ERROR] This test does not exists for the model and dataset.")
   
if __name__ == "__main__":
    main()

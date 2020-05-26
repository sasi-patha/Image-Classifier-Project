#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Sasi Kiran Patha
# DATE CREATED: 19-May-2020                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module for train and predict
#          of Image Classifier. If the user fails to provide some or all of the
#          inputs, then the default values are used for the missing inputs.
#          Refer to the functions below for for details about the arguments.
#          

# Imports python modules
import argparse
import os
from torchvision import models
import torch
from PIL import Image


def data_dir(data_directory):
    """
    Check if the directory exists and contains subfolders train, 
    valid and test.
    """
    
    data_directory = data_directory.strip()
#     data_directory = os.path.abspath(data_directory)
    
    if os.path.isdir(data_directory):
        pass
    elif os.path.isdir(os.path.abspath(
        data_directory)):
        data_directory = os.path.abspath(data_directory)            
    else:
        raise argparse.ArgumentTypeError(
            "Argument data_directory: {} is not a valid "
             "directory path.".format(data_directory))
    
    subdirs= []
    subdirs = filter(lambda x: os.path.isdir(os.path.join(data_directory, x)), 
                          os.listdir(data_directory))
    subdirs = list(subdirs)
        
    if ('train' in subdirs and 'valid' in subdirs and 'test' in subdirs):                 
        pass
    else:
        raise argparse.ArgumentTypeError(
            "Did not find some or all of the sub folders: 'train', 'valid', "
            "'test' in the data directory : {}".format(data_directory))
   
    return data_directory


def is_dir(save_directory):
    """
    Check if the directory exists.
    """
    
    save_directory = save_directory.strip()
    
    if os.path.isdir(save_directory):
        pass
    elif os.path.isdir(os.path.abspath(
        save_directory)):
        save_directory = os.path.abspath(save_directory)           
    else:
        raise argparse.ArgumentTypeError(
            "Argument save_directory: {} is not a valid "
            "directory path.".format(save_directory))
    
    return save_directory
        
    
def lr(learning_rate):
    """
    Check range of learning rate.
    """
    
    try: 
        learning_rate = float(learning_rate)
    except:
        raise argparse.ArgumentTypeError(
            "Argument learning_rate: {} is not a valid float, "
            "must be 0.001<=x<=0.005.".format(learning_rate))  
    
    if learning_rate > 0.005 or learning_rate < 0.001:
        raise argparse.ArgumentTypeError(
            "Argument learning_rate: {} is not a valid learning_rate, "
            "must be 0.001<=x<=0.005.".format(learning_rate))    
    
    return learning_rate


def drpout(dropout):
    """
    Check range of learning rate.
    """
    
    try: 
        dropout = float(dropout)
    except:
        raise argparse.ArgumentTypeError(
            "Argument dropout: {} is not a valid float, "
            "must be must be 0.3<=x<=0.7.".format(dropout))  
    
    if dropout > 0.7 or dropout < 0.3:
        raise argparse.ArgumentTypeError(
            "Argument dropout: {} is not a valid dropout, "
            "must be 0.3 <= x <= 0.7.".format(dropout))    
    
    return dropout


def check_gpu(gpu):
    """
    Validate if GPU is available.
    """
    
    if gpu and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("GPU is not available for execution "
             "using the option --gpu.") 
    
    return gpu
    
    
def parse_args_train():
    """
    Retrieves and parses the command line arguments provided by the user
    they run the train from a terminal window. 
    This function uses Python's argparse module to accept the command
    line arguments. 
    If the user fails to provide some or all of the arguments, then the
    default values are used for the missing arguments.
    
    Type python cmdline -h for details about cmd line arguments.
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command 
            line arguments
    Returns:
     parse_args() - data structure that stores the command line 
                    arguments object  
    """
    
    parser = argparse.ArgumentParser(description="Get inputs for "
                                     "Training Image Classifier.")
    
    parser.add_argument('data_directory', type=data_dir, 
                        help="Provide a directory with Images in "
                              "subfolders train, valid, test.")
    
    parser.add_argument('--save_dir', type=is_dir,
                    help="Provide a directory for storing model checkpoint.")
        
    arch_list =  list(filter(lambda x: '__' not in x ,models.__dict__.keys()))
    help_msg = ("Provide a valid model arch name available from Pytorch "
                "torchvision models.")
    parser.add_argument('--arch', type=str, default='resnet101',
                       help=help_msg, choices = arch_list)
        
    parser.add_argument('--learning_rate', type=lr, default=0.001,
                        help="Provide learning rate for model training "
                              "Gradient descent step. 0.001 <= x <= 0.005.")
    
    parser.add_argument('--hidden_units', type=int, nargs='+',default=[512],
                       help="eg: specify: 1024 512, for 2 hidden layers "
                        "with 1024 and 512 nodes.")
    
    parser.add_argument('--num_classes', type=int, default=102,
                        help="Provide Num of classes to train the model.")
    
    parser.add_argument('--epochs', type=int, default=5,
                        help="Provide num of epochs to train the model.")
    
    parser.add_argument('--gpu', action='store_true',
                       help="Provide if training should be done on gpu.")
    
    help_msg= ("Provide dropout percentage rate for model training. "
               "0.3 <= x <= 0.7.")
    parser.add_argument('--dropout', type=lr, default=0.30,
                        help=help_msg)
    
    help_msg = ("Provide a checkpoint file path for loading model "
                "checkpoint and restart training.")
    parser.add_argument('--checkpoint', type=chkp,
                       help=help_msg)
    
    hidden_activation_list = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU', 
                              'Hardtanh', 'Tanh']
    parser.add_argument('--hidden_activation', type=str, 
                        default='ReLU', choices=hidden_activation_list,
                        help="Provide a hidden layers activation function "
                        "for nn training.")
    
    output_activation_list = ['LogSoftmax', 'Softmax', 'Softplus']
    parser.add_argument('--output_activation', type=str, 
                        default='LogSoftmax', choices=output_activation_list,
                        help="Provide a output layers activation function "
                             "for nn training.")

    train_criterion_list = ['NLLLoss', 'CrossEntropyLoss']
    parser.add_argument('--criterion', type=str, choices=train_criterion_list,
                        default='NLLLoss',
                        help="Provide a loss criteria functionfor nn "
                        "training.")
                              
    train_optimzr_list = ['Adam', 'SGD', 'lr_scheduler','SparseAdam',
                          'RMSprop', 'Adamax', 'Adagrad', 'Adadelta',
                          'ASGD']
    parser.add_argument('--optimizer', type=str, choices=train_optimzr_list,
                        default='Adam',
                       help="Provide a optimizer function for nn training.")
    
    help_msg = "Provide 3 batch sizes for data to: Train, Validation, Test. "
    
    parser.add_argument('--batch_sizes', type=int, nargs=3, 
                       default=[256,128,128], help=help_msg)
    
    args = parser.parse_args()
    for batch_size in args.batch_sizes:
        # https://stackoverflow.com/questions/57025836/how-to-check-
        # if-a-given-number-is-a-power-of-two-in-python
        if ((batch_size & (batch_size-1) == 0) and batch_size != 0 
            and batch_size > 63):
            pass
        else:
            raise argparse.ArgumentTypeError(
                "Argument batch_sizes: {} is not a valid, must be powers of "
                "2, min value - 64".
                format(args.batch_sizes))  
    
    check_gpu(args.gpu)
    
    # Confirm the Train Parameters from user and assist on the 
    # help for parameters.
    proceed = True
    print('\nParameters Applicable for Training:\n',args)
    if not args.save_dir:
        print("\nModel checkpoint save_dir is not provided !!")
    print('\nPlease press Y to proceed with training')
    confirm = input()
    if confirm.strip().lower() == 'y':
        pass
    else:
        proceed = False
        
    return args, proceed


def image(im_path):
    """
    Verify the path is of valid image
    """
    
    im_path = im_path.strip()
    
    if os.path.exists(im_path):
        pass
    elif os.path.exists(os.path.abspath(im_path)):
        im_path = os.path.abspath(im_path)
    else:
        raise argparse.ArgumentTypeError(
            "Argument image file path : {} is not a valid "
            "file path".format(im_path))
    
    return im_path


def chkp(checkpoint_file_path):
    """
    Verify the checkpoint file
    """
    
    checkpoint_file_path = checkpoint_file_path.strip()
    
    if os.path.exists(checkpoint_file_path):
        pass
    elif os.path.exists(os.path.abspath(checkpoint_file_path)):
        checkpoint_file_path = os.path.abspath(checkpoint_file_path)
    else:
        raise argparse.ArgumentTypeError(
            "Argument checkpoint file path : {} is not a valid "
            "file path".format(checkpoint_file_path))
     
    
    return checkpoint_file_path


def is_catg(category_names):
    """
    Check is category to class label file mapping.
    """
    
    category_names = category_names.strip()
    
    if os.path.exists(category_names):
        pass
    elif os.path.exists(os.path.abspath(category_names)):
        category_names = os.path.abspath(category_names)
    else:
        raise argparse.ArgumentTypeError(
            "Argument category names file path : {} is not a valid "
            "file path.".format(category_names))
    
    return category_names
             
    
def parse_args_predict():
    """
    Retrieves and parses the command line arguments provided by the 
    user when they run the predict from a terminal window.
    This function uses Python's argparse module to accept the
    command line arguments. 
    If the user fails to provide some or all of the arguments, then
    the default values are used for the missing arguments.
    
    Type python cmdline -h for details about cmd line arguments.
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command 
            line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments 
                    object  
    """
    
    parser = argparse.ArgumentParser(description="Get inputs for Predict  "
                                    "using Image Classifier checkpoint model.")
    parser.add_argument('image_path', type=image,
                        help="Provide Image file path for which Name and "
                        "Class Probability should be predicted.")
    
    parser.add_argument('checkpoint', type=chkp,
                       help="Provide a checkpoint file path for loading "
                        "model checkpoint for prediction.")
    
    parser.add_argument('--top_k', type=int, default=1, 
                       help="Provide how many top k predictions are "
                        "needed. --top_k 3")
    
    parser.add_argument('--gpu', action='store_true',
                       help="Provide '--gpu' if predictions should be made "
                        "on gpu .")
    
    parser.add_argument('--category_names', type=is_catg,
                       help="Provide category to name json file path")
    
    args = parser.parse_args()
    
    check_gpu(args.gpu)
    
    # Confirm the Predict Parameters from user and assist on the help 
    # for parameters.
    proceed = True
    print('\nParameters Applicable for Predict:\n',args)
    print('\nPlease press Y to proceed with training')
    confirm = input()
    if confirm.strip().lower() == 'y':
        pass
    else:
        proceed = False
        
    return args, proceed


if __name__ == "__main__":
    
    args_train, proceed = parse_args_train()
    print(args_train)
    
#     args_predict, proceed = parse_args_predict()
#     print(args_predict)
#     check_gpu(args_predict.gpu)
    
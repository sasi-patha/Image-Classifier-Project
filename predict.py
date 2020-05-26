#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Sasi Kiran Patha
# DATE CREATED: 19-May-2020                                  
# REVISED DATE: 
# PURPOSE: Predict using the custom Image classifier which is built
#          using pre trained Pytorch Neural Network by applying
#          Transfer Learning.

import time
import logging
import torch

from cmdline import parse_args_predict
from data_utils import get_image, process_image, load_checkpoint, get_catg_name
from workspace_utils import active_session


def predict(image_torch_tensor, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep 
    learning model.
    """
    
    print("Predict the Image class starts...")
    
    model.to(device)   # Move the model to the available device.
    
    # Prediction.
    with torch.no_grad():
        model.eval()
        image_torch_tensor = image_torch_tensor.to(device)
        logps = model(image_torch_tensor)
        ps = torch.exp(logps)
        top_prob, top_indx = ps.topk(topk,dim=1)
    model.train()

    probs = top_prob.tolist()[0]
    indexes = top_indx.tolist()[0]
    class_to_idx_inv = {v: k for k, v in model.class_to_idx.items()} 
    classes = [class_to_idx_inv[indx] for indx in indexes]
      
    return probs, classes


def main():
    """
    Main Function for predictions using Image classifier on a 
    image path.
    Inputs are passed using command line.
    Type python predict.py -h for help on usage.
    """
    
    # Parse and validate the cmdline arguments passed for Training.
    args, proceed = parse_args_predict()
    
    if proceed:
        print('args in main:', args)
#         print('time for gpu')
#         return [], [], [] 
    else:
        print("\n Type python predict.py -h for help on setting training "
        "parameters. Thank you. \n")
        return [], [], [] 

    # Check if cuda is available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.gpu:
        device = torch.device("cpu")
    else:
        print("cuda is available? :", torch.cuda.is_available())
        raise Exception('Error ! device CUDA is not available')
    
    # Read Image data
    with active_session():
        image = get_image(args.image_path)
    # Pre process the image into numpy ndarray
        image_nparray = process_image(image)
    print("Pre Processing the Image complete.")
    # Convert numpy ndarray to torch tensor.
    image_torch_tensor = torch.from_numpy(image_nparray)
    
    # Assign first dimension of tensor for number of images = 1
    # as input to model.
    image_torch_tensor = image_torch_tensor.unsqueeze(0)
    image_torch_tensor = image_torch_tensor.float()
    
    probs, classes, class_labels = [], [], []
    # Load the model from checkpoint, Predict for input image.   
    with active_session():
        model, checkpoint = load_checkpoint(args.checkpoint)
        print("Loading the checkpoint complete.")
        probs, classes = predict(image_torch_tensor, model, device, 
                                 topk=args.top_k)
    
    print("Predict the Image class complete.")
    print ('Top {} predictions Probabilities: '.format(args.top_k), probs, 
               '\nTop {} predictions Classes: '.format(args.top_k), classes)
    
    if args.category_names:
        print("Mapping the Prediction class categories to class labels.")
        catg_to_name = get_catg_name(args.category_names)
        class_labels = [catg_to_name[x] for x in classes]
        print('Top {} predictions Class Labels: '.format(args.top_k), 
              class_labels)
    
    return probs, classes, class_labels
    
    
if __name__ == "__main__":
    
    """
    Predict App for classifying Images using deep learning trained 
    custom classifier ontop pytorch pre trained models.. 
    Type python predict.py -h for help on usage.
    """
    
    start = time.time()
    
    try:
        probs, classes, class_labels = main()
    except Exception as e:
        print(e)
        logging.exception(e)
        
    time_elapsed = time.time() - start
    print("\nTotal Predict session time: {:.0f}m {:.0f}s.\n".format( 
                                     time_elapsed//60, time_elapsed % 60))

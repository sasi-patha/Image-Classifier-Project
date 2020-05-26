#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Sasi Kiran Patha
# DATE CREATED: 19-May-2020                                  
# REVISED DATE: 
# PURPOSE: Data utils - pre processing, post processing, loading,
#          saving data required for the project.

import os
import json
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms

from network import build_network


def load_train_data(data_dir, batch_sizes):
    """
    Load the images data required for training, validation and testing.
    """
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir =  os.path.join(data_dir, 'valid')
    test_dir =  os.path.join(data_dir, 'test')
    
    print('\nTrain dir: {}, Validation dir: {}, Test dir: {}.\n'.format(
          train_dir,valid_dir,test_dir))
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    data_transforms = {'train': train_transforms, 'test': test_transforms, 
                       'valid': valid_transforms}

    # DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    image_datasets = {'train': train_data, 'test': test_data, 
                      'valid': valid_data}
    class_to_idx = image_datasets['train'].class_to_idx
    
    # DONE: Using the image datasets and the trainforms, define the dataloaders
    train_batch_size = batch_sizes[0]
    valid_batch_size = batch_sizes[1]
    test_batch_size = batch_sizes[2]
    
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],  
                                    batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'],  
                                    batch_size=test_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], 
                                    batch_size=valid_batch_size, shuffle=True)

    dataloaders = {'train': train_loader, 'test': test_loader, 
                   'valid': valid_loader}
    
    return dataloaders, class_to_idx


def load_checkpoint(checkpoint_filepath):
    """
    Load the model checkpoint
    """
    
    print("Loading the checkpoint...")
    
    if torch.cuda.is_available():
        #from https://github.com/pytorch/pytorch/issues/9139
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    # Note: build_network function(above in the begining) definition 
    # should be loaded / imported. build_network is stored
    #  as a checkpoint variable. Else torch.load fails for unpickle error.
    checkpoint = torch.load(checkpoint_filepath, map_location=map_location)
    
    func = checkpoint['build_nw_func']
    
    model = func(checkpoint['arch_name'], checkpoint['layers'], 
                 checkpoint['dropout'], checkpoint['hidden_activation'],
                 checkpoint['output_activation'])
    
    model.load_state_dict(checkpoint['state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint
    

def checkpoint_model(checkpoint, checkpoint_filepath):
    """
    Save the model checkpoint.
    """
    print('checkpoint_filepath: ',checkpoint_filepath)
    torch.save(checkpoint, checkpoint_filepath)
    print ('Checkpoint saved in the path.')
    return True


def get_image(image_path):
    """
    Return the image from image path.
    Check if image contains valid image data
    """
    
    image = Image.open(image_path)
    
    try:       
        im =Image.open(image_path)
        im.crop()
    except Exception as e:
        print(e)
        raise Exception("Image path: {} does not contain valid ime data.".
                        format(image_path))
    
    return image


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    
    # Process a PIL image for use in a PyTorch model
    print("Pre Processing the Image...")
    # Resize the images where the shortest side is 256 pixels, 
    # keeping the aspect ratio.
    image_size = image.size
    image_width, image_height = image_size
    print('Original Image dim: ', image_width, image_height)
    if image_width < image_height:
        resize_width = 256
        resize_dim =  [resize_width, image_height]
    else:
        resize_height = 256
        resize_dim = [image_width, resize_height]
    
    image.thumbnail(resize_dim)
    
    # Crop out the center 224x224 portion of the image.
    image_size = image.size
    image_width, image_height = image_size
    print('Thumbnail Image short 256 dim: ', image_width, image_height)
    crop_width = image_width - 224
    left_crop_location = crop_width / 2
    right_crop_location = left_crop_location + 224
    crop_height = image_height - 224
    top_crop_location = crop_height / 2
    bottom_crop_location = top_crop_location + 224
    crop_dim = (left_crop_location,top_crop_location,
                right_crop_location,bottom_crop_location)
    image_center_crop = image.crop(crop_dim)
    print('Center crop 224 Image dim: ', image_center_crop.size,'\n')
    
    # Convert Color channels of images from [0,255] to [0,1] - normalize.
    image_nparray = np.asarray(image_center_crop)
    image_nparray = image_nparray.astype('float32')
    image_nparray /= 255
    
    # The network expects the images to be normalized in a specific way 
    # - standardize.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    image_nparray = (image_nparray - means) / stds
    
    # PyTorch expects the color channel to be the first dimension.
    image_nparray = np.transpose(image_nparray, (2,0,1))
        
    return image_nparray


def get_catg_name(category_path):
    """
    Get mapping for category to class labels / names
    """
    
    with open(category_path, 'r') as ctg_names:
        catg_to_name  = json.load(ctg_names)
    
    return catg_to_name
    

if __name__ == "__main__":
    
    dataloaders = load_train_data(os.path.abspath('flowers'))
    images, labels = next(iter(dataloaders['train']))
    print(images[0:])
    
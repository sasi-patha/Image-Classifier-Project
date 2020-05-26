#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Sasi Kiran Patha
# DATE CREATED: 19-May-2020                                  
# REVISED DATE: 
# PURPOSE: Build pytorch pre trained model using meta programing
#          by using the parameters provided by user.


from torchvision import models
from torch import nn
from collections import OrderedDict


def build_network(arch_name='resnet101',layers=[512,102], dropout=0.3,
                hidden_activation='ReLU', output_activation='LogSoftmax'):
    """
    Builds the pre-trained torchvision network. 
    Freezes the pretrained network parameters.
    Add untrained FF network as final layer of the network, with input 
    Layers, drop out rate, output activation function.
    
    Inputs:
    -----------
    arch_name          : Name of the pre-trained torchvision network.
    ='resnet101'
    layers             : Number nodes in each hidden layer for untrained FF
    =[512, 102]          network.
                         Last entry is for output layer of the project.
                         Need not specify entry for pre-trained network's 
                         final layer, it is computed from the network.
    dropout            : Hidden layers ouput drop out probability.
    >= 0.0 & <1.0
    hidden_activation  : Activtation function for output of untrained FF.
      ='ReLU'
    output_activation  : Activtation function for output of untrained FF.
      ='LogSoftmax'
    
    Outputs:
    -----------   
    model              : Custom pre trained network with untrained FF n/w
                         ready  for training.
                        
    """
    
    # Check if a valid arch_name is passed.
    if not hasattr(models, arch_name):
        raise Exception('Invalid model : {}'.format(arch_name))  
    
    # Check if type of data passed for layers and drop out.
    all_num = filter(lambda x: isinstance(x, int) and x > 1, layers)
    if (not all_num or not isinstance(dropout,float)
        or dropout > 0.7 or dropout < 0.0):
        raise Exception("Please check the data types and values of "
                        "parameter layers - integers > 0, dropout - float")
    
    # Create the pre trained arch_name
    model = getattr(models, arch_name)(pretrained=True)
    
    # Get the input features size of the last layer of pre trained arch_name.
    try:
        last_layer_input_size = list(model.children())[-1].in_features
    except:
        last_layer_input_size = list(model.children())[-1][0].in_features
    
    # Combine the input feature size as first layer for untrained FF network.
    layersFF = [last_layer_input_size] + layers
    
    # Get the last child name of pre trained network so as to assign the
    # untrained FF network.
    last_layer_name = list(model.named_children())[-1][0]

    # Freeze parameters of pre trained arch_name - no backprop through them.
    for param in model.parameters():
        param.requires_grad = False
    
    # Build the sequential layers for untrained FF.
    fc_layers = OrderedDict()
    lh = len(layersFF)
    for i, layer in enumerate(layersFF):
        key_lin = 'F' + str(i+1)
        key_drop = 'D'  + str(i+1)
        key_act = 'A' + str(i+1)
        if (i+1 < lh and i+2 == lh):
            fc_layers[key_lin] = nn.Linear(layersFF[i], layersFF[i+1])
            fc_layers['output'] = getattr(nn, output_activation)(dim=1)
            break
        else:
            fc_layers[key_lin] = nn.Linear(layersFF[i], layersFF[i+1])
            fc_layers[key_drop] = nn.Dropout(p=dropout)
            fc_layers[key_act] = getattr(nn, hidden_activation)()
    
    # Attach the untrained FF to the final layer of pre trained network.
    model._modules[last_layer_name] = nn.Sequential(fc_layers)
    
    model.arch_name = arch_name
    model.layers = layers
    model.dropout = dropout
    model.hidden_activation = hidden_activation
    model.output_activation = output_activation
    model.last_layer_name = last_layer_name
    model.func = build_network
    
    return model


if __name__ == "__main__":
    
    model = build_network(arch_name='resnet101',layers=[512,102], dropout=0.3,
                    hidden_activation='ReLU', output_activation='LogSoftmax')
    
    print(model)
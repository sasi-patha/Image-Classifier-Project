
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Sasi Kiran Patha
# DATE CREATED: 19-May-2020                                  
# REVISED DATE: 
# PURPOSE: Training a custom Image classifier using pre trained Pytorch 
#          Neural Network by applying Transfer Learning.


import time
import os
import logging
import torch
from torch import nn, optim

from cmdline import parse_args_train, check_gpu
from data_utils import load_train_data, load_checkpoint, checkpoint_model
from network import build_network
from workspace_utils import active_session


def train_network(model, device = 'cpu', 
                  criteria='NLLLoss', optimizr='Adam', learn_rate=0.003, 
                  trainloader=[], validloader = [],  
                  epochs=3, checkpoint={}):
    """
    Trains a network custom torch model network by name.
    Early Stops training if 2 consecutive epocs validation losses increases.
    Resumes Model Training.
    Accumulates epoch Losses and Accuracies when the training the resumes
    the parameters.
    
    Inputs:
    -----------
    model        :  Custom pre trained network to train.
    device       :  Device to execute the model Training.
    = 'cpu'/'cuda'
    criteria     :  Loss Function for back propogation.
    optimizr     :  optimizer for gradient descent.
    learn_rate   :  learning rate for the optimizer, scaling factor for 
                    gradient descent step.
    trainloader  :  Train set of images loaded with ImageFolder and
                    transformations made.
    validloader  :  Validation set of images loaded with ImageFolder and
                    transformations made.         
    last_epoch   :  0 implies initial training, else training 
                    resumes from the epoch.
    epochs       :  Number of epochs to train.
    checkpoint   :  Should be Empty for initial Train.
                    Else contains checkpoint values for:
                    a. Accumulated Losses and Accuracies from previous 
                       training for the model and any tests.
                    b. last epoch
                    c. checkpoint data.
                    
    Outputs:
    -----------
    checkpoint      :  Returns checkpoint values for:
                       a. Accumulated Losses and Accuracies from previous 
                          training for the model and any tests.
                       b. last epoch.
                       c. checkpoint data.
    model           :  Returns the Trained Model (best based on Early Stop)
    
    """
    
    # Check if datasets are populated for training.
    if not trainloader or not validloader:
        raise Exception('Error ! check train loader and valid loader'
                        'contains data')
    
    # Check learn_rate values
    if not isinstance(learn_rate, float) or learn_rate <= 0.0:
        raise Exception('Provide right value for learn_rate,'
                       'ideally around 0.0002')
    
    # Check last_epoch, epochs
    last_epoch = checkpoint.get('last_epoch',0)
    if (not isinstance(last_epoch, int) or not isinstance(epochs, int)
        or last_epoch <0 or epochs <0):
        raise Exception('Provide right values for last_epoch, epochs')
    
    # Check the cuda is available
    if device.type == 'cuda' and torch.cuda.is_available():
        pass
    elif device.type == 'cpu':
        pass
    else:
        print(device.type, torch.cuda.is_available())
        raise Exception('Error ! device CUDA is not available')
    
    # Set the optimizer and criterion
    last_layer_name = list(model.named_children())[-1][0]
    optimizer = getattr(optim, optimizr)(model._modules[last_layer_name].parameters(),
                                         lr=learn_rate)
    criterion = getattr(nn, criteria)()
    

    # These Quantities are accumalated accross the epochs and training sessions.
    if not last_epoch and checkpoint.get('train', False):
        raise Exception("last epoch in checkpoint is 0, Model is trained.")
    elif not last_epoch and not checkpoint.get('train', False):
        train_losses, valid_losses, accuracies = [], [], []
    else:
        if ( not checkpoint.get('train',False) or
             not checkpoint.get('valid',False) or
             not checkpoint.get('accuracies',False) ):
            raise Exception("Missing checkpoint losses for resume training")
        else:
            print('Loading optimizer for Training from Checkpoint')
            train_losses = checkpoint['train']
            valid_losses = checkpoint['valid']
            accuracies = checkpoint['accuracies']
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    model.to(device)    # Move the model to the available device.
    
    epoch_idx = 0
    
    print('Training Started....')
    
    for e in range(last_epoch+1, last_epoch + epochs + 1 ):
        e_start = time.time()
        epoch_train_loss = 0
        
        if (epoch_idx > 2 and 
            valid_losses[epoch_idx-1] > valid_losses[epoch_idx-2] and
            valid_losses[epoch_idx-2] > valid_losses[epoch_idx-3]):
            # Early stopping if 2 consecutive epochs have increase in 
            # validation loss.
            break
        
        train_pass = 0
        for images, labels in trainloader:
#             start = time.time()
            train_pass += 1
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if train_pass % 3 == 0:
                print("Train Step No: {}..".format(train_pass),
                   "Step Train loss: {:.4f}.".format(loss.item()))
#                 time_elapsed = time.time() - start
#                 print("Step Training time: {:.0f}m {:.0f}s".format(
#                                             time_elapsed//60, 
#                                             time_elapsed % 60))

        epoch_accuracy = 0.0
        epoch_valid_loss = 0.0
        valid_pass = 0
        with torch.no_grad():
            model.eval()
            for imagesv, labelsv in validloader:
#                 start = time.time()
                valid_pass += 1
                imagesv = imagesv.to(device)
                labelsv = labelsv.to(device)
                logps = model(imagesv)
                valid_loss = criterion(logps, labelsv)
                ps = torch.exp(logps)
                top_prob, top_class = ps.topk(1,dim=1)
                equals = top_class == labelsv.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                epoch_accuracy += accuracy.item()*100
                epoch_valid_loss += valid_loss.item()
                if valid_pass % 3 == 0:
                    print("Valid Step No: {}.. ".format(valid_pass),
                    "Step Valid loss: {:.4f}, ".format(valid_loss.item()),
                    "Step Valid Accuracy: {:.4f} %. ".format(
                                                     accuracy.item()*100))
#                     time_elapsed = time.time() - start
#                     print("Step Valid time: {:.0f}m {:.0f}s".format(
#                                                        time_elapsed//60, 
#                                                        time_elapsed % 60))
        model.train()
        epoch_train_loss = epoch_train_loss/len(trainloader)
        train_losses.append(epoch_train_loss)
        epoch_valid_loss = epoch_valid_loss / len(validloader)
        valid_losses.append(epoch_valid_loss)
        epoch_accuracy = epoch_accuracy / len(validloader)
        accuracies.append(epoch_accuracy)
        print("Epoch No.: {}/{} --".format(e,last_epoch + epochs),
              "Epoch Training loss: {:.4f}, ".format(epoch_train_loss),
              "Epoch Valid loss: {:.4f}, ".format(epoch_valid_loss),
              "Epoch Accuracy: {:.4f} %.".format(epoch_accuracy))
        e_time_elapsed = time.time() - e_start
        print("Epoch time: {:.0f}m {:.0f}s.\n".format(e_time_elapsed//60, 
                                                    e_time_elapsed % 60))
            
        epoch_idx +=1
    
#     plt.xlim(1, len(valid_losses) + 1)
#     plt.plot(train_losses, color='c', label = 'Train Losses')
#     plt.plot(valid_losses, color='r', label = 'Valid Losses')
#     plt.xlabel('Epochs')
#     plt.ylabel('Losses') 
#     plt.title('Train vs Validation Losses')
#     plt.legend() 
#     plt.show()
    
    checkpoint['train'] = train_losses
    checkpoint['valid'] = valid_losses
    checkpoint['accuracies'] = accuracies
    checkpoint['last_epoch'] = last_epoch + epochs
    checkpoint['arch_name'] = model.arch_name
    checkpoint['layers'] = model.layers
    checkpoint['dropout'] = model.dropout
    checkpoint['hidden_activation'] = model.hidden_activation
    checkpoint['output_activation'] = model.output_activation
    checkpoint['build_nw_func'] = model.func
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['criteria'] = criterion
    
    return model, checkpoint

            
def main():
    """
    Main Function for Training a Image classifier on a directory
    containing sub folders train, valid , test with Images.
    Inputs are passed using command line.
    Type python train.py -h for help on usage.
    """
    
    # Parse and validate the cmdline arguments passed for Training.
    args, proceed = parse_args_train()
    
    if proceed:
        print('args in main:', args)
    else:
        print("\n Type python train.py -h for help on setting training "
                "parameters. Thank you! \n")
        return True
    
    # Load the Training data 
    dataloaders, class_to_idx = load_train_data(args.data_directory, 
                                                args.batch_sizes)
    
    # Build network
    layers= args.hidden_units + [args.num_classes]
    model = build_network(arch_name=args.arch, layers=layers, 
                          dropout=args.dropout,
                          hidden_activation=args.hidden_activation, 
                          output_activation=args.output_activation)
    
    # Check if cuda is available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.gpu:
        device = torch.device("cpu")
    else:
        print("cuda is available? :", torch.cuda.is_available())
        raise Exception('Error ! device CUDA is not available')
        
    # load the model from checkpoint, checkpoint for losses and 
    # model params for training restart.
    with active_session():
        if args.checkpoint:
            model, checkpoint = load_checkpoint(args.checkpoint)
        else:
            checkpoint = {}
        
        trained_model, checkpoint = train_network(model, device,
                                              criteria=args.criterion, 
                                              optimizr=args.optimizer,
                                              learn_rate=args.learning_rate,
                                              trainloader=dataloaders['train'], 
                                              validloader=dataloaders['valid'],
                                              epochs=args.epochs, 
                                              checkpoint=checkpoint)
        if args.save_dir:
            checkpoint['class_to_idx'] = class_to_idx
            checkpoint_filename = ""
            checkpoint_filename += 'checkpoint' 
            checkpoint_filename +=  str(checkpoint['last_epoch'])
            checkpoint_filename += '.pth'        
            checkpoint_filepath = os.path.join(args.save_dir, checkpoint_filename)
            checkpoint_model(checkpoint, checkpoint_filepath)
    

if __name__ == "__main__":
    """
    Train App for training a image classifier model using deep learning
    with pytorch pre trained models.
    Type python train.py -h for help on usage.
    """
        
    start = time.time()
    
    try:
        main()
    except Exception as e:
        print(e)
        logging.exception(e)
        
    time_elapsed = time.time() - start
    print("\nTotal Training session time: {:.0f}m {:.0f}s.\n".format(time_elapsed//60, 
                                                      time_elapsed % 60))
# -*- coding: utf-8 -*-
"""
Last modified: 17Apr (Leo)

"""
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

# Import modules
"""
Please make sure data_process_classification.py, 
neural_network_model_classification.py & utils.py 
is in the same folder as this file
"""
from neural_network_model_classification import NeuralNetworkModel
import utils as utils

# Display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set random seed for consistency
seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Use device: ", device)

#code_path = [insert path]
#save_path = [insert path]
#result_path = [insert path]

# Load datasets
print("Reading train_dataset...")
train_dataset = pd.read_csv(code_path + 'train_dataset.csv')
print("Reading val_dataset...")
val_dataset = pd.read_csv(code_path + 'val_dataset.csv')
print("Finished reading datasets.")

# Create categorical Y columns, separating numerical Y into classes
Y_cols = ['useful', 'funny', 'cool']

num_classes = 3         # Define number of classes

# In main function - create one-hot Y's 
train_dataset = utils.create_y_categories(train_dataset, Y_cols, num_classes)
val_dataset = utils.create_y_categories(val_dataset, Y_cols, num_classes)

# In main function - create new Y column name lists
useful_Y_cols = utils.gen_new_Y_cols_list('useful', num_classes)
funny_Y_cols = utils.gen_new_Y_cols_list('funny', num_classes)
cool_Y_cols = utils.gen_new_Y_cols_list('cool', num_classes)

# Drop the old Y columns at the end
train_dataset = train_dataset.drop(columns=Y_cols)
val_dataset = val_dataset.drop(columns=Y_cols)

# Define the non-text Y's
X_nontext_cols = [i for i in train_dataset if
                  i not in useful_Y_cols + funny_Y_cols + cool_Y_cols + ['text']]

# For DistilBertTokenizer, ['CLS']: 101, ['SEP']: 102

# Initialize hyperparameters
max_seq_len = 256
batch_size = 64
n_epochs = 5
lr = 5e-4
weight_decay = 1e-5
early_stop = 2

# Generate dataloaders
print("Generating train_dataloader...")
train_dataloader = utils.gen_dataloader(train_dataset, 'train', X_nontext_cols, useful_Y_cols, 
                   funny_Y_cols, cool_Y_cols, max_seq_len, batch_size)
print("Generating val_dataloader...")
val_dataloader = utils.gen_dataloader(val_dataset, 'val', X_nontext_cols, useful_Y_cols, 
                   funny_Y_cols, cool_Y_cols, max_seq_len, batch_size)
print("Finished generating dataloaders.")

########### Training process ###########
# Train models for each Y SEPARATELY
for Y in Y_cols:
    
    # Create neural network model, optimizer, loss function
    model = NeuralNetworkModel(input_dim=len(X_nontext_cols), 
                               output_dim=num_classes,
                               max_seq_len=max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Initialize variables
    iter_wout_improve = 0  # Early stop
    max_acc = 0
    train_loss_over_epoch = []
    train_acc_over_epoch = []
    val_loss_over_epoch = []
    val_acc_over_epoch = []
    val_f1_scores = []
    
    print(f"Start training {Y} Model")
    for epoch in range(n_epochs):
    
        # Train mode
        model.train()
        train_loss = []
        train_accs = []
        print(f"Running epoch {epoch+1} of {n_epochs} for {Y} model:")
        for batch in tqdm(train_dataloader, desc="Training..."):
            # collect data
            non_text_data, text_data, text_mask, useful_target, funny_target, cool_target = batch
            # calculate predicted probablity
            probs = model(non_text_data.to(device), text_data.to(device), text_mask.to(device))
            # Pick the current Y, calculate loss, and optimize
            if Y == 'useful':
                target = useful_target
            elif Y == 'funny':
                target = funny_target
            elif Y == 'cool':
                target = cool_target
            
            loss = criterion(probs, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = torch.max(probs, dim=1)[1].to('cpu')
            label = torch.max(target, dim=1)[1]
            acc = torch.sum(pred == label).item() / len(pred)
            
            # log accuracy, loss
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        # Calculate average train loss & acc
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
    
        ## Record training loss and accuracy in each epoch
        train_loss_over_epoch.append(train_loss)
        train_acc_over_epoch.append(train_acc)
        print("Training epoch {} of {} : loss = {:4f}, acc = {:4f}".format(epoch + 1, n_epochs, train_loss, train_acc))
   
        # Validate mode
        model.eval()
        val_loss = []
        val_accs = []
        all_preds = []
        all_labels = []
        for batch in tqdm(val_dataloader, desc="Validating"):
            # collect data
            non_text_data, text_data, text_mask, useful_target, funny_target, cool_target = batch
            # calculate predicted probablity
            with torch.no_grad():
                probs = model(non_text_data.to(device), text_data.to(device), text_mask.to(device))
            # Pick the current Y, calculate loss
            if Y == 'useful':
                target = useful_target
            elif Y == 'funny':
                target = funny_target
            elif Y == 'cool':
                target = cool_target
            
            loss = criterion(probs, target.to(device))
            
            # Calculate acc
            pred = torch.max(probs, dim=1)[1].to('cpu')
            label = torch.max(target, dim=1)[1]
            acc = torch.sum(pred == label).item() / len(pred)
            
            # log accuracy, loss
            val_loss.append(loss.item())
            val_accs.append(acc)
            all_preds += pred.tolist()
            all_labels += label.tolist()
        
        # Calculate average val loss and acc
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = sum(val_accs) / len(val_accs)
        
        # Record average validation loss and accuracy in each epoch
        val_loss_over_epoch.append(val_loss)
        val_acc_over_epoch.append(val_acc)
        
        # Get F1-score of ALL labels in [cls0, cls1, ...]
        val_f1_score = f1_score(y_true=all_labels, y_pred=all_preds, labels=list(range(num_classes)), average=None).tolist()
        val_f1_scores.append(val_f1_score)
        
        # Print the information
        print("Validating epoch {} of {} : loss ={:4f}, acc = {:4f}".format(epoch + 1, n_epochs, val_loss, val_acc))
        print("F1-scores:", val_f1_score)
        
        # Save model if reached a higher validation acc
        if val_acc > max_acc:
            max_acc = val_acc
            print('\nSaving model (epoch = {:03d}, acc = {:.4f})'.format(epoch + 1, max_acc))
            torch.save(model.state_dict(), save_path + f'{Y}_model.pth')
            iter_wout_improve = 0
        else:
            iter_wout_improve += 1
    
        # Early stop mechanism to shorten training time
        if iter_wout_improve == early_stop:
            print(f'Early stopping since validation loss did not improve for {iter_wout_improve} epochs.')
            break
    
    # Free memory for next model training
    del model
    torch.cuda.empty_cache()

    # Turn recorded validation f1_scores into arrays
    val_f1_scores = np.array(val_f1_scores)
    
    # Plot graph after training
    utils.plot_graph(Y, "loss", train_loss_over_epoch, val_loss_over_epoch, graph_path)
    utils.plot_graph(Y, "accuracy", train_acc_over_epoch, val_acc_over_epoch, graph_path)
    utils.plot_f1_scores(Y, "F1-score", val_f1_scores, num_classes, graph_path)
    
    # Save the records into a CSV
    output_df = pd.DataFrame({"train_loss": train_loss_over_epoch,
                              "val_loss": val_loss_over_epoch,
                              "train_acc": train_acc_over_epoch,
                              "val_acc": val_acc_over_epoch
                              })
    for i in range(num_classes):
        colname = f"f1_score_class{i}"
        output_df[colname] = val_f1_scores[:,i]
    
    result_pathname = result_path + Y + ".csv"
    output_df.to_csv(result_pathname, index=False)
    
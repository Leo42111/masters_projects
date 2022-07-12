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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import modules
"""
Please make sure data_process_classification.py, 
neural_network_model_classification.py & utils.py 
is in the same folder as this file
"""
from data_process_classification import Dataprocess
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
#device = 'cpu'
print("Use device: ", device)

#code_path = [insert path]
#save_path = [insert path]
#result_path = [insert path]

# Read test dataset
print("Reading test_dataset...")
test_dataset = pd.read_csv(code_path+'test_dataset.csv')
print("Finished loading test_dataset.")

# Create categorical Y columns, separating numerical Y into classes
Y_cols = ['useful', 'funny', 'cool']

num_classes = 3         # Define number of classes

## In main function - create one-hot Y's 
test_dataset = utils.create_y_categories(test_dataset, Y_cols, num_classes)

## In main function - create new Y column name lists
useful_Y_cols = utils.gen_new_Y_cols_list('useful', num_classes)
funny_Y_cols = utils.gen_new_Y_cols_list('funny', num_classes)
cool_Y_cols = utils.gen_new_Y_cols_list('cool', num_classes)

test_dataset = test_dataset.drop(columns=Y_cols)

# Define the non-text Y's
X_nontext_cols = [i for i in test_dataset if
                  i not in useful_Y_cols + funny_Y_cols + cool_Y_cols + ['text']]

# Initialize hyperparameters
max_seq_len = 256
batch_size = 64

print("Generating test_dataloader...")

# Create Dataprocess (Dataset) class using the dataset
test_dataprocess = Dataprocess(test_dataset, 
                               X_nontext_cols, 
                               useful_Y_cols, 
                               funny_Y_cols, 
                               cool_Y_cols,
                               max_seq_len=max_seq_len)

# Create dataloader using the created Dataprocess class
test_dataloader = DataLoader(test_dataprocess,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)

print("Finished generating test_dataloader.")

# Initialize empty DataFrame for later record stacking
all_results = pd.DataFrame({"Y": [],
                            "test_loss": [],
                            "test_acc": [],
                            "test_f1_scores": [],
                            })

# Loop through each response
for Y in Y_cols:

    # Create models and load saved parameters
    print(f"Loading saved {Y} classification model...")
    
    model = NeuralNetworkModel(input_dim=len(X_nontext_cols),
                               output_dim=num_classes,
                               max_seq_len=max_seq_len).to(device)
    
    model.load_state_dict(torch.load(save_path+Y+"_model.pth", map_location='cpu'))
    
    print("Finished loading saved models.")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Prepare log lists for classification
    test_loss_log = []
    test_accs_log = []
    all_preds = []
    all_labels = []
    model.eval()
    
    # Feed the test data into classification model first
    for batch in tqdm(test_dataloader, desc=f"Evaluating test data on {Y} model"):
        
        # collect data
        non_text_data, text_data, text_mask, useful_target, funny_target, cool_target = batch
        
        # Transfer the data to device
        non_text_data = non_text_data.to(device)
        text_data = text_data.to(device)
        text_mask = text_mask.to(device)
        useful_target = useful_target.to(device)
        funny_target = funny_target.to(device)
        cool_target = cool_target.to(device)
    
        # calculate predicted probablity
        with torch.no_grad():
            probs = model(non_text_data.to(device), text_data.to(device), text_mask.to(device))
            
        # calculate loss, and optimize
        if Y == 'useful':
            target = useful_target
        elif Y == 'funny':
            target = funny_target
        elif Y == 'cool':
            target = cool_target
            
        # calculate loss
        loss = criterion(probs, target.to(device))
        
        # Calculate acc
        pred = torch.max(probs, dim=1)[1]
        label = torch.max(target, dim=1)[1]
        acc = torch.sum(pred == label).item() / len(pred)
            
        # log and calculate accuracy, loss
        test_loss_log.append(loss.item())
        test_accs_log.append(acc)
        all_preds += pred.tolist()
        all_labels += label.tolist()
        
    # Calculate average test loss and acc
    test_loss = sum(test_loss_log) / len(test_loss_log)
    test_acc = sum(test_accs_log) / len(test_accs_log)
    
    # Get F1-score of ALL labels in [cls0, cls1, ...]
    test_f1_score = f1_score(y_true=all_labels, y_pred=all_preds, labels=list(range(num_classes)), average=None).tolist()
    
    # Log the results into the all_results DataFrame
    results = pd.DataFrame({"Y": Y,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "test_f1_scores": test_f1_score,
                            })
    all_results = all_results.append(results, ignore_index=True)

    # Print all evaluation values at the end
    print(f"Size of test data:   {len(test_dataset)}")
    print(f"Classification for {Y}: test loss = {test_loss:4f}, test acc = {test_acc:4f}")
    print("F1-scores:", test_f1_score)

# Save the results to a csv
# NOTE: the f1-scores will be split into rows
all_results.to_csv(result_path+"test_classification_results.csv", index=False)

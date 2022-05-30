# -*- coding: utf-8 -*-
"""
Last modified: 17Apr (Leo Ma)

"""

# Import libraries
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_process_classification import Dataprocess
from torch.utils.data import DataLoader

# Map function to apply in create_y_categories()
# Classify the Y into classes (numeric)
def classify_y(x):
    if x <= 1:
        return 0    # Class 0
    elif x <= 4:  
        return 1    # Class 1
    return 2        # Class 2

# Map function to apply in create_y_categories()
# Create one-hot labels for one column
def one_hot_encode(x, i):
    if x == i:
        return 1
    else:
        return 0

# Function to create categorical y based on numerical Y
def create_y_categories(dataset, regression_Y_cols, num_classes):
    
    # Loop through each regression Y
    for Y in tqdm(regression_Y_cols, desc="One-hot encoding each Y"):  
        
        colname = f"{Y}_class"
        # Use classify_y() to map Y into different classes
        dataset[colname] = dataset[Y].apply(classify_y)
        
        # Manual one-hot encoding of the new colname column
        for i in range(num_classes):
            new_colname = f"{colname}_{i}"
            dataset[new_colname] = dataset[colname].apply(one_hot_encode, args=(i,))
            
        # Drop the newly created "colname" column
        dataset = dataset.drop(columns=[colname])
        
    return dataset

# Function to generate lists of new column names of Y, based on passesd "attr"
def gen_new_Y_cols_list(attr, num_classes):
    output_list = []
    for i in range(num_classes):
        new_colname = f"{attr}_class_{i}"
        output_list.append(new_colname)
    return output_list


# Function to generate dataloader
# mode: 'train': enable shuffling in DataLoader, 'val': disable shuffling
def gen_dataloader(dataset, mode, X_nontext_cols, useful_Y_cols, 
                   funny_Y_cols, cool_Y_cols, max_seq_len, batch_size):
    # Create Dataprocess (Dataset) class using the dataset
    data = Dataprocess(dataset, X_nontext_cols, useful_Y_cols, 
                       funny_Y_cols, cool_Y_cols, max_seq_len)

    # Create DataLoader using the created dataset class & return it
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=(mode == 'train'), drop_last=True)
    return dataloader

"""
# Function to plot graph for easier model evaluation
# Y: "useful" / "funny" / "cool"
# metric: "loss" / "accuracy"
"""

def plot_graph(Y, metric, train_list, val_list, graph_path):
    plt.figure(figsize=(10, 10))
    plt.plot(train_list, color='blue', label='train')
    plt.plot(val_list, color='red', label='val')
    plt.title(f"Train and validation {metric} for {Y} model")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    pathname = graph_path + Y + "_" + metric + ".png"
    plt.savefig(pathname)


# Function to plot F1-scores of each class
def plot_f1_scores(Y, metric, f1_scores, num_classes, graph_path):
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap("tab10")
    for i in range(num_classes):
        plt.plot(f1_scores[:,i], color=cmap(i), label=f'Class_{i}')
    plt.title(f"F1-scores of each class in {Y} model")
    plt.xlabel("Epochs")
    plt.ylabel("F1-scores")
    plt.legend()
    pathname = graph_path + Y + "_" + metric + ".png"
    plt.savefig(pathname)
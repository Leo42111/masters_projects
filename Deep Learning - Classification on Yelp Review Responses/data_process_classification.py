# -*- coding: utf-8 -*-
"""
Last modified: 17Apr (Leo Ma)

"""

# Import libraries
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


# Custom Dataset class to pre-process before creating DataLoader
class Dataprocess(Dataset):
    
    # Initialization function when initializing a Dataprocess object
    def __init__(self, data, X_nontext_cols, useful_Y_cols, 
                 funny_Y_cols, cool_Y_cols, max_seq_len):
        # Extract non_text predictors, review text & each kind of response
        nontext_data = data[X_nontext_cols]
        text_data = data['text'].tolist()
        useful_target = data[useful_Y_cols]
        funny_target = data[funny_Y_cols]
        cool_target = data[cool_Y_cols]
        
        # Create tensor for nontext X
        self.nontext_data = torch.FloatTensor(torch.from_numpy(np.array(nontext_data).astype(np.float32)))
        
        # Handle review text X, and create tokenized text data & mask
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("Loaded pretrained DistilBertTokenizer. Starting apply tokenizer to test_data")
        text_temp = tokenizer(text_data,  # Tokenizer will tokenize lists of texts & return a large tensor
                                max_length=max_seq_len,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')

        print("Loaded tokenizer to test_data")


        self.text_data = text_temp["input_ids"]
        self.text_mask = text_temp["attention_mask"]
        
        # Create tensors for Y's
        self.useful_target = torch.FloatTensor(useful_target.to_numpy())
        self.funny_target = torch.FloatTensor(funny_target.to_numpy())
        self.cool_target = torch.FloatTensor(cool_target.to_numpy())

    # Return these 5 items in a batch of DataLoader
    def __getitem__(self, index):
        return self.nontext_data[index], self.text_data[index], self.text_mask[index], \
            self.useful_target[index], self.funny_target[index], self.cool_target[index]

    # Returns the length of the dataset
    def __len__(self):
        return len(self.nontext_data)

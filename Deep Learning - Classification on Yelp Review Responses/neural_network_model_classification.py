# -*- coding: utf-8 -*-
"""
Last modified: 17Apr (Leo Ma)

"""

# Import libraries
import torch
import torch.nn as nn
from transformers import DistilBertModel

# NN Model class
class NeuralNetworkModel(nn.Module):

    # Initialization when creating a class object
    # input_dim: nontext dimension; output_dim = num_classes, max_seq_len: max sequence length
    def __init__(self, input_dim, output_dim, max_seq_len):
        super(NeuralNetworkModel, self).__init__()

        # Feed-forward part for non-text X
        self.nontext_ff = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )

        # NLP model for text X - pre-trained DistilBert
        # dimension for DistilBert = 768
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Top layer for DistilBert model
        self.text_model_top_ff = nn.Sequential(
            nn.Linear(max_seq_len * 768, 2048),
            nn.Sigmoid()
        )

        # Feed-forward part after combining hidden layers from non_text X and text X
        # Merged final_ff to here
        self.combined_ff = nn.Sequential(
            nn.Linear(512+2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(1024, output_dim),
            nn.Softmax(dim=1)
        )

    # Forward function when X is pass to the model
    def forward(self, nontext, text, mask):
        nontext_out = self.nontext_ff(nontext)
        # nontext_out = (batch_size, 512)
        text_out = self.text_model(text, mask)
        # text_out = (batch_size, max_seq_len, hidden_size=768)
        text_out = text_out['last_hidden_state'].view(text.size(0), -1)  ### Fixed hard code batch size
        # text_out = (batch_size, (max_seq_len*hidden_size)=196608)
        text_out = self.text_model_top_ff(text_out)
        # text_out = (batch_size, 2048)
        # Concatenate the hidden output of non-text & text part
        combined_out = torch.cat((nontext_out, text_out), dim=1)
        # combined_out = (batch_size, 512+2048=2560)
        combined_out = self.combined_ff(combined_out)
        # combined_out = (batch_size, output_dim)

        return combined_out
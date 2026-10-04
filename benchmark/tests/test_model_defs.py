"""
Tests deep learning models in models/model_defs.py
"""

import pytest 

import torch
from benchmark.models.model_defs import ESM_seperateCNN, AAindex_seperateCNN, AAindex_pca_seperateCNN, OHE_seperateCNN

# testing the data flow before the model is most important 

# check model dimensions 
# input dimensions 
# internal dimensions 
# output dimensions 

# check compatibility with dataloaders 


class Test_ESM_separateCNN:

    # pytest specifically looks for method `setup_class` in the test class 
    def setup_method(self):
        self.model = ESM_seperateCNN()
        self.x1 = torch.rand(size=(1,1,12,320))
        self.x2 = torch.rand(size=(1,1,48,320))
        self.x = [self.x1, self.x2]
    
    def test_cnn1_output_shape(self):
        assert self.model.cnn1(self.x1).shape == (1,128)

    def test_cnn2_output_shape(self):
        assert self.model.cnn2(self.x2).shape == (1,128)
    
    def test_output_shape(self):
        assert self.model(self.x).shape == (1,1)


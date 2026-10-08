"""
Tests deep learning models in models/model_defs.py
"""

import pytest 

import torch
from benchmark.models.model_defs import ESM_separateCNN, AAindex_separateCNN, AAindex_pca_separateCNN, OHE_separateCNN

class Test_ESM_separateCNN:
 
    def setup_method(self):
        self.model = ESM_separateCNN()
        self.x1 = torch.rand(size=(1,1,12,320))
        self.x2 = torch.rand(size=(1,1,48,320))
        self.x = [self.x1, self.x2]
    
    def test_cnn1_output_shape(self):
        assert self.model.cnn1(self.x1).shape == (1,128)

    def test_cnn2_output_shape(self):
        assert self.model.cnn2(self.x2).shape == (1,128)
    
    def test_output_shape(self):
        assert self.model(self.x).shape == (1,1)


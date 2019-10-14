import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import Isomorphic_Feature_Extraction, Classification_Component


class IsoNN(nn.Module):
  
    def __init__(self, k, c, feature_size, hidden1_size, hidden2_size, num_classes):
     
        super(IsoNN, self).__init__()
        self.feature_network = Isomorphic_Feature_Extraction(k, c)
        self.classifier = Classification_Component(feature_size, hidden1_size, hidden2_size, num_classes)



    def forward(self, x, last=True):

        features = self.feature_network(x)
        log_probas = self.classifier(features)
        return log_probas

  
import torch
from torch import nn
import numpy as np


class EmotionQuantizer(nn.Module):
    def __init__(self, bins_file):
        super().__init__()
        self.ADV_bins = np.load(bins_file)
        self.arousal_bins = len(self.ADV_bins['arousal_bin'])
        self.dominance_bins = len(self.ADV_bins['dominance_bin'])
        self.valence_bins = len(self.ADV_bins['valence_bin'])
        
    def dynamic_quantize(self, values, device):
        """Nonlinear quantization converts ADV values into ADV tokens"""
        # dynamically create container boundaries (automatically align devices with each forward propagation)
        arousal_bins = torch.from_numpy(self.ADV_bins['arousal_bin']).to(device)
        dominance_bins = torch.from_numpy(self.ADV_bins['dominance_bin']).to(device)
        valence_bins = torch.from_numpy(self.ADV_bins['valence_bin']).to(device)
        
        # quantization
        tokens = torch.zeros_like(values, dtype=torch.int32, device=device)
        tokens[:, 0] = torch.bucketize(values[:, 0].contiguous(), arousal_bins, right=True)
        tokens[:, 1] = torch.bucketize(values[:, 1].contiguous(), dominance_bins, right=True)
        tokens[:, 2] = torch.bucketize(values[:, 2].contiguous(), valence_bins, right=True)
        
        # boundary protection
        min_indices = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        max_indices = torch.tensor([self.arousal_bins, self.dominance_bins, self.valence_bins], 
                                   dtype=torch.int32, device=device) - 1
        return torch.clamp(tokens, min=min_indices, max=max_indices)
    
    def inverse_quantize(self, tokens, device):
        """Reverse map ADV_token to ADV continuous values and take the median of adjacent box boundaries"""
        
        arousal_bins = torch.from_numpy(self.ADV_bins['arousal_bin']).to(device)
        dominance_bins = torch.from_numpy(self.ADV_bins['dominance_bin']).to(device)
        valence_bins = torch.from_numpy(self.ADV_bins['valence_bin']).to(device)
        tokens = torch.clamp(tokens, 
                        min=torch.tensor([0, 0, 0], dtype=torch.int32, device=device), 
                        max=torch.tensor([self.arousal_bins, self.dominance_bins, self.valence_bins], 
                                         dtype=torch.int32, device=device) - 1)
        # median
        a_left = arousal_bins[tokens[:, 0] - 1]
        a_right = arousal_bins[tokens[:, 0]]
        a_values = (a_left + a_right) / 2
        
        d_left = dominance_bins[tokens[:, 1] - 1]
        d_right = dominance_bins[tokens[:, 1]]
        d_values = (d_left + d_right) / 2
        
        v_left = valence_bins[tokens[:, 1] - 1]
        v_right = valence_bins[tokens[:, 1]]
        v_values = (v_left + v_right) / 2
        
        ADV = torch.stack([a_values, d_values, v_values], dim=1)
        return ADV.to(torch.float32)
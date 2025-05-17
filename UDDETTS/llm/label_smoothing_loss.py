"""Label smoothing module."""

import torch
from torch import nn    

class EmoAwareLabelSmoothingLoss(nn.Module):
    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        super(EmoAwareLabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length
        self.emo_weight = 5.0

    def forward(self, x: torch.Tensor, target: torch.Tensor, emo_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
            
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        
        # Generate emo position weight matrix
        emo_weights = torch.ones_like(target, dtype=torch.float)
        emo_mask = torch.zeros_like(target, dtype=torch.bool)
        for b in range(batch_size):
            emo_label_pos = emo_positions[b]
            emo_weights[b, emo_label_pos] = self.emo_weight  # Enlarge the weight of emo label position
            emo_mask[b, emo_label_pos] = True  # Mark the position of the emo label

        # Flattening
        x = x.view(-1, self.size) # (B*T, V)
        target = target.view(-1) # (B*T,)
        emo_weights = emo_weights.view(-1) # (B*T,)
        emo_mask = emo_mask.view(-1) # (B*T,)
        
        # Standard label smoothing calculation
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B*T,) The position of the ignored label (-1) in the target
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence) # one-hot
        
        # KL loss
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        # Loss weighting
        weighted_kl = kl.sum(dim=1) * emo_weights
        denom = total if self.normalize_length else batch_size
        loss = weighted_kl.masked_fill(ignore, 0).sum() / denom
        # emotion label loss(may be zero because of some emotion labels masked)
        emo_loss = kl.sum(dim=1).masked_fill(ignore, 0)[emo_mask]
        emo_loss = emo_loss[emo_loss != 0]
        if emo_loss.numel() > 0:
            emo_loss = emo_loss.mean()
        else:
            emo_loss = torch.tensor(0.0).to(emo_loss.device)
            
        return loss, emo_loss

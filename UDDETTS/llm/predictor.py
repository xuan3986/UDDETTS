from transformers import RobertaModel
import torch
import torch.nn as nn

class ADVPredictor(nn.Module):
    def __init__(self, pretrained_model, text_token_size):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        for para in self.roberta.parameters():
            para.requires_grad = True
        self.roberta.requires_grad_(True)        
        self.text_embedding = torch.nn.Embedding(text_token_size, self.roberta.config.hidden_size)
        # self.projection_lm = nn.Linear(self.roberta.config.hidden_size, text_token_size, bias=False)
        self.adv_head = nn.Linear(self.roberta.config.hidden_size, 3)
        self.activation = nn.Sigmoid()
        
    def forward(self, text_token, text_len):
        """
        input:
            text_token: [batch_size, max_seq_len]
            text_len: [batch_size]
        output:
            adv_values: [batch_size, 3]
        """
        batch_size, max_seq_len = text_token.size(0), text_token.size(1)
        text_emb = self.text_embedding(text_token)  # [batch_size, max_seq_len, hidden_size]
        positions = torch.arange(max_seq_len, device=text_emb.device).expand(batch_size, max_seq_len)
        attention_mask = (positions < text_len.unsqueeze(1)).float()
        
        lm_outputs = self.roberta(
            inputs_embeds = text_emb, 
            attention_mask= attention_mask,
            return_dict=False,
        )
        hidden_states, pooled_output = lm_outputs
        adv_values = self.adv_head(pooled_output)  # [batch_size, 3]
        adv_values = self.activation(adv_values) * 6.0 + 1.0
        return adv_values


class ADVLoss(nn.Module):
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, pred, target, bin_centers, mask):
        """
        input:
            pred: [batch_size, 3]
            target: [batch_size, 3]
            mask: [batch_size, ]
        output:
            loss: [1, ]
        """
        pred = pred[mask]
        target = target[mask]
        bin_centers = bin_centers[mask]
        loss1 = self.mse(pred, target).sum()
        loss2 = self.mse(pred, bin_centers).sum()
        loss = self.alpha * loss1 + loss2
        return loss
        
        


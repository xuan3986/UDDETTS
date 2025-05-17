import torch
from torch import nn

class LMADVEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # independent encoders for each ADV dimension
        self.arousal_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )
        self.dominance_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )
        self.valence_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )
        
        # Cross attention pooling layer (compresses 3 features into 1)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=4,  # or 3
            batch_first=True
        )
        # Learnable query vectors (for attention)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, adv, mask):
        """
        input: adv - [batch, 3] (arousal, dominance, valence)
        output: [batch, 3, embed_dim]
        """
        a = adv[:, 0].unsqueeze(-1)  # arousal
        d = adv[:, 1].unsqueeze(-1)  # dominance
        v = adv[:, 2].unsqueeze(-1)  # valence
        a_emb = self.arousal_encoder(a)  # [batch, embed_dim]
        d_emb = self.dominance_encoder(d)
        v_emb = self.valence_encoder(v)
        adv_emb = torch.stack([a_emb, d_emb, v_emb], dim=1)
        
        query = self.query.repeat(adv_emb.size(0), 1, 1)  # [batch, 1, embed_dim]
        attn_output, _ = self.cross_attn(
            query=query, 
            key=adv_emb, 
            value=adv_emb
        )  # attn_output: [batch, 1, embed_dim]
        # The part where mask ADV is [0,0,0]
        attn_output[~mask] = 0
        return attn_output
import torch
import torch.nn as nn

class EmoMixtureEncoder(nn.Module):
    def __init__(self, num_emotions, arousal_bins, dominance_bins, valence_bins, embed_dim):
        super().__init__()
        # Emotion label embedding
        self.emo_embed = nn.Sequential(
            nn.Embedding(num_emotions, 120),
            nn.LayerNorm(120)
        )
        # ADV 3D embedding
        self.arousal_embed = nn.Embedding(arousal_bins, 120//3)
        self.dominance_embed = nn.Embedding(dominance_bins, 120//3)
        self.valence_embed = nn.Embedding(valence_bins, 120//3)
        self.null_adv = nn.Parameter(torch.zeros(1, 120))
        # ADV interaction layer
        self.adv_interaction = nn.Sequential(
            nn.Linear(120, 240),
            nn.GELU(),
            nn.Linear(240, 120),
            nn.LayerNorm(120)
        )
        # Cross modal fusion layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=120, 
            num_heads=4,
            dropout = 0.1,
            batch_first=True
        )
        # mapping layer
        self.attn_proj = nn.Sequential(
            nn.Linear(120, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.emo_proj = nn.Sequential(
            nn.Linear(120, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.adv_proj = nn.Sequential(
            nn.Linear(120, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # gating layer
        self.gate = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def get_adv_emb(self, adv_token):
        a_emb = self.arousal_embed(adv_token[:,0])
        d_emb = self.dominance_embed(adv_token[:,1])
        v_emb = self.valence_embed(adv_token[:,2])
        adv_emb = torch.cat([a_emb, d_emb, v_emb], dim=-1)
        return adv_emb

    def forward(self, emo_token, adv_token, adv_mask, feat_len):
        """
        input:   emo_token - [batch, ] (emotion_idx)
                adv_tokens - [batch, 3] (arousal_idx, dominance_idx, valence_idx)
        output:   [batch, feat_len, embed_dim]
        """
        emo_emb = self.emo_embed(emo_token) # [B, ] -> [B,120]
        adv_emb = torch.where(
            adv_mask.unsqueeze(1),  # [B,1]
            self.get_adv_emb(adv_token),  # validation sample
            self.null_adv.expand_as(emo_emb)  # invalid sample
        )
        adv_emb = self.adv_interaction(adv_emb).unsqueeze(1) # [B,1,120]
        emo_emb = emo_emb.unsqueeze(1)
        # Multi head attention
        attn_output, _ = self.cross_attn(
            query=emo_emb, 
            key=adv_emb,
            value=adv_emb
        ) # attn_output: [B,1,120]

        attn_output = self.attn_proj(attn_output) # [B,1,embed_dim]
        emo_emb = self.emo_proj(emo_emb) # [B,1,embed_dim]
        adv_emb = self.adv_proj(adv_emb) # [B,1,embed_dim]
        # semi-supervised gating algorithm
        gate = self.gate(torch.cat([emo_emb, attn_output], dim=-1))
        mask_factor = adv_mask.float().view(-1,1,1)  # [B,1,1]
        gate_output = (gate + (1 - mask_factor)) * emo_emb + (1 - gate) * mask_factor * attn_output
        gate_output = gate_output.repeat(1, feat_len, 1) # [B,feat_len,embed_dim]
        adv_emb_expanded = adv_emb.repeat(1, feat_len, 1)  # [B,feat_len,embed_dim]
        emo_mask = (emo_token == 0).view(-1, 1, 1).expand_as(gate_output)
        output = torch.where(emo_mask, adv_emb_expanded, gate_output)
        return output
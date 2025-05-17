from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from encoder import EmoMixtureEncoder
from UDDETTS.llm.quantizer import EmotionQuantizer
from UDDETTS.utils.mask import make_pad_mask
    

class DiffWithADVLabel(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,  # llm speech token size
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 ADV_bins_file: str = "",
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        
        self.quantizer = EmotionQuantizer(ADV_bins_file)
        self.ADV_encoder = EmoMixtureEncoder(
            num_emotions=10, # emo label size = 10
            arousal_bins=self.quantizer.arousal_bins,
            dominance_bins=self.quantizer.dominance_bins,
            valence_bins=self.quantizer.valence_bins,
            embed_dim = output_size
        )
        self.grad_stats = {}
        
    def _grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            # Monitor output gradient
            for g_out in grad_output:
                if g_out is not None:
                    output_grads = g_out.norm().item()
                    if torch.isnan(g_out).any(): print(f"module {name} grad_output has nan")
            # Monitor input gradient
            for g_in in grad_input:
                if g_in is not None:
                    input_grads = g_in.norm().item()
                    if torch.isnan(g_in).any(): print(f"module {name} grad_input has nan")
            self.grad_stats[name] = { 
                'output_grads': output_grads,
                'input_grads': input_grads
            }
        return hook

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        modules_to_monitor = {
            "ADV_encoder": self.ADV_encoder,
            "spk_embedding": self.spk_embed_affine_layer,
            "speech_embedding": self.input_embedding,
            "flow_encoder": self.encoder,
            "length_regulator": self.length_regulator,
        }
        handles = []
        for name, module in modules_to_monitor.items():
            handle = module.register_backward_hook(self._grad_hook(name))
            handles.append(handle)
            
        
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        spk_embedding = batch['embedding'].to(device)
        ADV = batch['ADV'].to(device)
        emo_token = batch['emo_id'].to(device)
  
        # speaker
        spk_embedding = F.normalize(spk_embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(spk_embedding) # [B, dim]
        
        # Quantify ADV and encode it with emo token as conditions
        ADV_token = self.quantizer.dynamic_quantize(ADV, device)
        ADV_mask = (ADV_token != 0).any(dim=1)  # If ADV_token is [0,0,0], then the corresponding position of the mask is False
        conds = self.ADV_encoder(emo_token, ADV_token, ADV_mask, feat.shape[1]) # [B, feat_len, dim]
        
        # speech semantic token
        mask = (~make_pad_mask(speech_token_len)).float().unsqueeze(-1).to(device)
        speech_token_embedding = self.input_embedding(torch.clamp(speech_token, min=0)) * mask
        h, h_lengths = self.encoder(speech_token_embedding, speech_token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            spk_embedding,
            cond=conds.transpose(1, 2).contiguous()
        )
        for handle in handles:
            handle.remove()
            
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  ADV_token,
                  emo_token,
                  spk_embedding,
                  flow_cache):
        # speaker
        spk_embedding = F.normalize(spk_embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(spk_embedding)

        # speech token
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(spk_embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len = int(token_len / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h, mel_len, self.input_frame_rate)

        # ADV conds
        ADV_mask = (ADV_token != 0).any(dim=1)
        conds = self.ADV_encoder(emo_token, ADV_token, ADV_mask, mel_len)
   
        mask = (~make_pad_mask(torch.tensor([mel_len]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=spk_embedding,
            cond=conds.transpose(1, 2).contiguous(),
            n_timesteps=10,  # diffusion time steps
            flow_cache=flow_cache
        )
        assert feat.shape[2] == mel_len
        return feat.float(), flow_cache

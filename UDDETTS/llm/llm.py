from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from quantizer import EmotionQuantizer
from predictor import ADVPredictor, ADVLoss
from adv_encoder import LMADVEncoder
from label_smoothing_loss import EmoAwareLabelSmoothingLoss
from UDDETTS.utils.common import IGNORE_ID, th_accuracy


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
            ADV_bins_file: str = "",
            roberta_file: str = "",
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear( 
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build language model related modules
        self.sos_eos = 0
        self.attribute_id = 1
        self.task_id = 2
        self.llm_embedding = torch.nn.Embedding(3, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 2)
        self.loss_llm = EmoAwareLabelSmoothingLoss(
            size=speech_token_size + 2,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.loss_adv = ADVLoss()
        
        # 3. build ADV label related modules
        self.quantizer = EmotionQuantizer(ADV_bins_file)
        self.ADV_predictor = ADVPredictor(roberta_file, text_token_size)
        self.ADV_encoder = LMADVEncoder(llm_input_size) # can be added to enhance the text embedding's emotional expression
        self.ADV_embedding = torch.nn.Embedding(
            max(self.quantizer.arousal_bins,
                self.quantizer.dominance_bins,
                self.quantizer.valence_bins),
            llm_input_size)
        self.emo_embedding = torch.nn.Embedding(10, llm_input_size) # emo label size = 10
        
        # 4. build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 5. sampling method
        self.sampling = sampling
        
        # 6. gradient monitor
        self.grad_stats = {}

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_embedding, text_token_embedding, text_token_len, 
                           attribute_embedding, spk_embedding, ADV_token_embedding,
                           task_id_embedding, emo_token_embedding, speech_token_embedding, speech_token_len):
        text_token_embedding = unpad_sequence(text_token_embedding, text_token_len.cpu(), batch_first=True)
        speech_token_embedding = unpad_sequence(speech_token_embedding, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([ 
                        sos_eos_embedding.squeeze(dim=0), text_token_embedding[i], 
                        attribute_embedding.squeeze(dim=0), spk_embedding[i], ADV_token_embedding[i], 
                        task_id_embedding.squeeze(dim=0), emo_token_embedding[i], speech_token_embedding[i]
                    ], dim=0) for i in range(len(text_token_embedding))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len
    
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
            # "ADV_encoder": self.ADV_encoder,
            "ADV_predictor": self.ADV_predictor,
            "llm": self.llm,
            "llm_decoder": self.llm_decoder,
            "ADV_embedding": self.ADV_embedding,
            "emo_embedding": self.emo_embedding
        }
        handles = []
        for name, module in modules_to_monitor.items():
            handle = module.register_backward_hook(self._grad_hook(name)) # register_backward_hook
            handles.append(handle)


        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        spk_embedding = batch['embedding'].to(device)
        ADV = batch['ADV'].to(device)
        emo_token = batch['emo_id'].to(device)
        emo_token = emo_token.unsqueeze(1)

            
        # 1. ADV to ADV_token and encode ADV_token emo_token, then mask
        ADV_token = self.quantizer.dynamic_quantize(ADV, device)
        mask = (ADV_token != 0).any(dim=1)  # If ADV_token is [0,0,0], then the corresponding position of the mask is False
        ADV_token_embedding = self.ADV_embedding(ADV_token)
        ADV_token_embedding[~mask] = 0 
        emo_token_embedding = self.emo_embedding(emo_token)
        # ADV_token_embedding: torch.Size([batch, 3, 1024]), emo_token_embedding: torch.Size([batch, 1, 1024])

        emo_parts = torch.stack([emo_token[i] for i in range(text_token.size(0))])
        emo_parts[~mask] = IGNORE_ID # Ignore ADV=[0,0,0]
        emo_parts[emo_parts == 0] = IGNORE_ID  # Ignore unknown labels

        # 2. prepare llm_target, EOS token=self.speech_token_size
        lm_target = [torch.tensor([IGNORE_ID] * (2 + 4 + text_token_len[i]) + emo_parts[i].tolist() + speech_token[i, :speech_token_len[i]].tolist() +
                                    [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)  
        emo_positions = 6 + text_token_len


        # 3. encode text_token
        text_token_embedding = self.text_embedding(text_token)
        text_token_embedding , text_token_len = self.encode(text_token_embedding , text_token_len) # text encoder
         # torch.Size([batch, text_token_len_max, 1024])
         
        # 4. predict ADV
        pred_ADV = self.ADV_predictor(text_token, text_token_len) # reberta encoder + regression head
        loss1 = self.loss_adv(pred_ADV, ADV, mask)
            
            
        # 5. add ADV_embedding to text_token_embedding
        # adv_emb = self.ADV_encoder(ADV, mask)
        # text_token_embedding = text_token_embedding + adv_emb  # adv_emb:torch.Size([batch, 1, 1024])

        # 6. speaker embedding projection
        spk_embedding = F.normalize(spk_embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(spk_embedding)
        spk_embedding = spk_embedding.unsqueeze(1)
        # torch.Size([batch, 1, 1024])

        # 7. sos attr and task_id
        sos_eos_embedding = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        attribute_embedding = self.llm_embedding.weight[self.attribute_id].reshape(1, 1, -1)
        task_id_embedding = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        # torch.Size([1, 1, 1024])

        # 8. speech_token
        speech_token_embedding = self.speech_embedding(speech_token)

        # 9. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_embedding, text_token_embedding , text_token_len, 
                                                        attribute_embedding, spk_embedding, ADV_token_embedding,
                                                        task_id_embedding, emo_token_embedding, speech_token_embedding, speech_token_len)

        # 10. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)

        # 11. loss and acc
        loss2, loss_emo = self.loss_llm(logits, lm_target, emo_positions)
        loss = loss1 + loss2
            
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 2), lm_target, ignore_label=IGNORE_ID)
        
        for handle in handles:
            handle.remove()
            
        return {
            'loss': loss,
            'loss_llm': loss2,
            'loss_adv': loss1,
            'loss_emo': loss_emo, 
            'acc': acc
        }

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            spk_embedding: torch.Tensor,
            ADV_token: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20, # can set to 30 if the synthesized tokens are not enough
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        # 1. ADV
        mask = (ADV_token != 0).any(dim=1)
        pred_ADV = self.ADV_predictor(text, text_len)
        pred_ADV_token = self.quantizer.dynamic_quantize(pred_ADV, device)
        filled_ADV_token = ADV_token.clone()
        filled_ADV_token[~mask] = pred_ADV_token[~mask]
        ADV_embedding = self.ADV_embedding(filled_ADV_token)
        
        
        # ADV = self.quantizer.inverse_quantize(ADV_token, device)
        # filled_ADV = ADV.clone()
        # filled_ADV[~mask] = pred_ADV[~mask]
        # adv_emb = self.ADV_encoder(filled_ADV, mask)
        
        # 2. text
        text = self.text_embedding(text)
        text, text_len = self.encode(text, text_len)
        # text = text + adv_emb

        # 3. spk_embedding
        if spk_embedding.shape[0] != 0:
            spk_embedding = F.normalize(spk_embedding, dim=1)
            spk_embedding = self.spk_embed_affine_layer(spk_embedding)
            spk_embedding = spk_embedding.unsqueeze(1)
        else:
            spk_embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 4. eos attr and task_id
        sos_eos_embedding = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        attribute_embedding = self.llm_embedding.weight[self.attribute_id].reshape(1, 1, -1)
        task_id_embedding = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_input = torch.concat([sos_eos_embedding, text, 
                                 attribute_embedding, spk_embedding, ADV_embedding, 
                                 task_id_embedding], dim=1)

        # 5. cal min/max_length
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # 6. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input, 
                offset=offset,
                required_cache_size=-1,
                att_cache=att_cache, 
                cnn_cache=cnn_cache,
                att_mask=torch.tril( # Causal Attention Mask
                    torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)
                ).to(torch.bool)
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # 7. generate tokens step by step
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), 
                out_tokens, 
                sampling, 
                ignore_eos=True if i < min_len else False
            ).item()
            # 8. detect EOS token
            if top_ids == self.speech_token_size:
                break
            # 9. Streaming output
            yield top_ids
            # 10. Status update
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            if offset < text_len + 7 + 1:
                lm_input = self.emo_embedding.weight[top_ids].reshape(1, 1, -1)
            else:
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

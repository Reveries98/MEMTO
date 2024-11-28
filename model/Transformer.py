import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from model.model.Transformer import EncoderLayer, Encoder
# ours
from .ours_memory_module import MemoryModule
# memae
# from .memae_memory_module import MemoryModule
# mnad
# from .mnad_memory_module import MemoryModule
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        logits = self.fc(x)  
        weights = F.softmax(logits, dim=-1)  
        return weights

class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # self.decoder_layer = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2,
        #                              batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(d_model, c_out)
        self.gate = GatingNetwork(input_dim=d_model,num_experts=2)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        # self.decoder_layer_add = nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        N,L,C = x.shape
        # out = self.decoder_layer1(x.transpose(-1, 1))
        # out = self.dropout(self.activation(self.batchnorm(out)))

        # decoder ablation
        # for _ in range(10):
        #     out = self.dropout(self.activation(self.decoder_layer_add(out)))

        # out = self.decoder_layer2(out).transpose(-1, 1)     
        '''
        out : reconstructed output
        '''

        x = x.reshape(N,L,2,-1)
        gate_weight = self.gate(x[:,:,0,:])
        out = self.out_linear(x)
        out = torch.sum(out*gate_weight.unsqueeze(-1),dim=-2)


        # out = self.out_linear(x)
        return out      # N x L x c_out


class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, win_size, enc_in, c_out, n_memory,configs = None, shrink_thres=0, backbone='Transformer', \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)   # N x L x C(=d_model)
        self.backbone = backbone
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        

        # ours
        self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)
        # self.weak_decoder = Decoder(2*d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        # baselines
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    
    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
        x = self.embedding(x)   # embeddin : N x L x C(=d_model)
        queries = out = self.encoder(x)   # encoder out : N x L x C(=d_model)
        
        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":queries, "mem":mem}
        else:
            
            out = self.weak_decoder(out)
            
            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out":out, "memory_item_embedding":memory_item_embedding, "queries":queries, "mem":mem, "attn":attn}

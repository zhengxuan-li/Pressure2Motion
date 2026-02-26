# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import os
import torchvision

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class Adapter(nn.Module):
    def __init__(self, encoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.pressure_stage2 = PositionalPressureEncoder(out_dim=256)

    def forward(self,
                hidden_states: Optional[Tensor] = None,
                text_embedding: Optional[Tensor] = None,
                pressure: Optional[Tensor] = None
                ):

        intermediate = []
        pressure_embedding = self.pressure_stage2(pressure)

        pressure_embedding = pressure_embedding.permute(1, 0, 2) # seq, batch, 256

        for i, layer in enumerate(self.layers):
            output = layer(hidden_states[i], text_embedding, pressure_embedding)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output

class AdapterLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                    activation="relu", normalize_before=False):
        super().__init__()
        # Cross-Attention for text and pressure embeddings
        self.cond_linear = nn.Linear(256, d_model) 
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)

        # Self-Attention and FFN from TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, text_embedding, pressure_embedding):
        # Cross-Attention block
        pressure_embedding = self.cond_linear(pressure_embedding) # seq, batch, d_model
        cond = torch.cat([text_embedding, pressure_embedding], dim=0) # seq+1, batch, d_model
        src2 = self.cross_attn(query=src, key=cond, value=cond)[0]
        src = src + self.dropout_cross(src2)
        src = self.norm_cross(src)

        # Self-Attention block
        q = k = self.with_pos_embed(src, None)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, text_embedding, pressure_embedding):
        # Cross-Attention block
        src2 = self.norm_cross(src)
        cond = torch.cat([text_embedding, pressure_embedding], dim=0)
        src2 = self.cross_attn(query=src2, key=cond, value=cond)[0]
        src = src + self.dropout_cross(src2)

        # Self-Attention block
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, None)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)

        # FFN block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, text_embedding, pressure_embedding):
        if self.normalize_before:
            return self.forward_pre(src, text_embedding, pressure_embedding)
        return self.forward_post(src, text_embedding, pressure_embedding)

class TransformerEncoder(nn.Module):
    # ok ignore norm + add norm--> layer norm
    def __init__(self, encoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                control: Optional[Tensor] = None):
        output = src
        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if control is not None:
                output = output + control[i]

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionalPressureEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        flattened_dim = 32 * 20 * 15
        self.fc = nn.Linear(flattened_dim, out_dim)

    def forward(self, pressure_seq):
        B, T, H, W = pressure_seq.shape

        pressure_norm = ((pressure_seq / 255.0) * 2.0 - 1.0)  # [B, T, 160, 120]
        pressure_diff = pressure_norm[:, 1:] - pressure_norm[:, :-1] # [B, T-1, 160, 120]

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        pos = torch.stack([xx, yy], dim=0).to(pressure_seq.device)
        pos = pos.unsqueeze(0).unsqueeze(0).repeat(B, T-1, 1, 1, 1)
        # -> pos: [B, T-1, 2, 160, 120]

        x = pressure_norm[:,:-1].unsqueeze(2)
        x = torch.cat([x, pressure_diff.unsqueeze(2), pos], dim=2)
        # -> x: [B, T-1, 4, 160, 120]

        x = x.view(B * (T-1), 4, H, W)
        # -> x: [B*(T-1), 4, 160, 120]

        feat = self.cnn(x)
        # -> feat: [B*(T-1), 32, 40, 30]

        feat_flat = feat.view(B, T-1, -1)
        # -> feat_flat: [B, T-1, 32 * 20 * 15] = [B, T-1, 9600]

        output = self.fc(feat_flat)


        return output
    
class Pressure(nn.Module):
    def __init__(self, pretrained=None):
        super(Pressure, self).__init__() 

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=False)
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1024)

        self.gru = TemporalEncoder(
            n_layers=2,
            input_size=1024,
            hidden_size=1024,
            bidirectional=True,
            add_linear=False,
            use_residual=True,
        )

        self.self_attention = nn.MultiheadAttention(1024, 4)
        self.regressor = Regressor(1024)

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask
    
    def forward(self, input):
        # input shape: [B, T, H, W]
        batchsize, windowSize, height, width = input.size()
        input = input.reshape(batchsize * windowSize, 1, height, width)

        x = self.resnet(input)  # [B*T, 1024]
        x = x.view(batchsize, windowSize, -1)  # [B, T, 1024]

        # GRU with residual
        res_x = x
        x = self.gru(x)
        x = x + res_x

        res_x = x
        x = self.gru(x)
        x = x + res_x

        # Self-Attention with residual
        res_x = x
        x, _ = self.self_attention(x, x, x)
        x = x + res_x
        
        x = x.reshape(-1, x.size(-1)) # [B*T, 1024]

        output = self.regressor(x)
        output = output.view(batchsize, windowSize, -1)

        return output

class Regressor(nn.Module):
    def __init__(self, feature_size):
        super(Regressor, self).__init__()

        self.fc1 = nn.Linear(feature_size, 1024)
        self.drop1 = nn.Dropout()
        self.decposition = nn.Linear(1024, 15)
        self.decrotation = nn.Linear(1024, 24)
        nn.init.xavier_uniform_(self.decposition.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decrotation.weight, gain=0.01)


    def forward(self, x, n_iter=1):

        for i in range(n_iter):
            xc = x.clone()
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            pred_position = self.decposition(xc)
            pred_rotation = self.decrotation(xc)

        output = torch.cat([pred_position, pred_rotation], 1)
        return output
 
class TemporalDownsampleEncoder(nn.Module):
    def __init__(self, input_emb_width=1024, output_emb_width=512, down_t=2, stride_t=2):
        super().__init__()
        layers = [
            nn.Conv1d(input_emb_width, 512, kernel_size=3, stride=stride_t, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, output_emb_width, kernel_size=3, stride=stride_t, padding=1),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B, T, C]
        x = x.transpose(1, 2)  # → [B, C, T]
        x = self.net(x)        # → [B, C_out, T_out]
        return x.transpose(1, 2)  # → [B, T_out, C_out]
    
class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            input_size=1024,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.input_size = input_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, input_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, input_size)
        self.use_residual = use_residual

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == self.input_size:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y
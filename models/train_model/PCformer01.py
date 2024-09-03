import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block3rand, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer4,TwoStageAttentionLayer2random
from models.PatchTST import FlattenHead
from layers.Autoformer_EncDec import series_decomp
from layers.Conv_Blocks import Inception_Block_V1



from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len,
                                                  self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
                      groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_seg_num, self.in_seg_num, kernel_size=3, stride=1, padding=1,
                      groups=self.in_seg_num),
            nn.BatchNorm2d(self.in_seg_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_seg_num, self.in_seg_num, kernel_size=1),
            nn.BatchNorm2d(self.in_seg_num),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv1d(in_channels=self.seq_len*2, out_channels=self.pred_len, kernel_size=1)

        self.softmax = nn.Softmax(-1)

        self.Linear0 = nn.Linear(self.seq_len-1, self.seq_len)
        self.Linear1 = nn.Linear(configs.d_model, self.seg_len)
        self.Linear2 = nn.Linear(self.seq_len, self.pred_len)




    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ### 将数据转化到频域空间
        # period, fx = FFT_for_Period(x_enc)
        x_orgin = x_enc
        B, T, N = x_enc.size()
        batch = x_enc.shape[0]
        ts_d = x_enc.shape[2]
        x_enc_diff = torch.diff(x_enc, n=1, dim=1)
        # 将差分结果添加回原始张量，以恢复原来的维度
        # x_enc = torch.cat((x_enc[:, :1, :], x_enc_diff), dim=1)
        # 在特定维度的第一个位置添加0

        x_enc = torch.nn.functional.pad(x_enc_diff, (0, 0, 1, 0, 0, 0), value=0)

        x_orgin = x_orgin - x_enc
        # x_enc = self.Linear0(x_enc)


        # mmm = x_enc[:, -1, :].unsqueeze(1)
        # x_enc = x_enc - mmm
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc = rearrange(x_enc, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        ##time
        # print('aa', time_in.shape)

        x_enc = x_enc.view(batch * ts_d, self.in_seg_num, 8, 64)
        x_enc = x_enc.permute(0, 2, 1, 3)
        # x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        dec_out1 = self.conv1(x_enc)  ### b d patch num patch_num

        #
        # dec_out_T = dec_out.permute(0, 1, 3, 2)
        #
        # V = torch.einsum("bhkd,bhdl->bhkl", dec_out, dec_out_T)
        #
        # V = self.softmax(V)
        # dec_out = torch.einsum("bhkd,bhdl->bhkl", V, dec_out) + dec_out

        dec_out1 = dec_out1.permute(0, 2, 1, 3)
        dec_out1 = self.norm2(dec_out1)

        dec_out1 = dec_out1.view(batch * ts_d, self.in_seg_num, 8*64)
        # dec_out2 = self.conv2(x_enc)
        #
        # dec_out2 = dec_out2.permute(0, 2, 1, 3)
        #
        # dec_out = torch.cat((dec_out1,dec_out2), dim=2)


        # dec_out = self.depthwise(x_enc)
        # dec_out = self.bn_depth(dec_out)
        # dec_out = self.relu(dec_out)
        #
        # dec_out = self.pointwise(dec_out)
        # dec_out = self.bn_point(dec_out)
        # dec_out = self.relu(dec_out)

        dec_out1 = rearrange(dec_out1, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)
        dec_out = self.Linear1(dec_out1)
        dec_out = dec_out.permute(0, 2, 3, 1).reshape(B, -1, N)
        # print('aa', dec_out.shape)
        # dec_out = self.output(dec_out)

        dec_out = self.Linear2(dec_out.permute(0, 2, 1))
        dec_out = dec_out.permute(0, 2, 1)
        # dec_out = dec_out + mmm
        # print(dec_out.shape)
        # dec_out = dec_out[:, :(self.seq_len + self.pred_len), :]

        # print('输出长度', dec_out.shape)
        return dec_out + x_orgin

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
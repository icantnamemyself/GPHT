import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GPHTBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()

        self.multipiler = configs.pred_len // configs.GT_patch_len[depth]
        self.down_sample = nn.AvgPool1d(
            self.multipiler) if configs.pooling == 'avg' else nn.MaxPool1d(self.multipiler)
        d_model = configs.GT_d_model
        self.patch_embedding = PatchEmbedding(d_model, configs.GT_patch_len[depth],
                                              configs.GT_patch_len[depth], 0, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model,
                        configs.n_heads),
                    d_model,
                    4 * d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    norm=configs.norm
                ) for l in range(configs.GT_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model) if configs.norm == 'ln' else torch.nn.BatchNorm1d(
                d_model)
        )
        self.forecast_head = nn.Linear(d_model, configs.pred_len)

    def forward(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        x_enc = self.down_sample(x_enc)
        enc_out, n_vars = self.patch_embedding.encode_patch(x_enc)
        enc_out = self.patch_embedding.pos_and_dropout(enc_out)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        bs = enc_out.shape[0]

        # Decoder
        dec_out = self.forecast_head(enc_out).reshape(bs, n_vars, -1)  # z: [bs x nvars x seq_len]
        return dec_out.permute(0, 2, 1)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs

        self.encoders = nn.ModuleList([
            GPHTBlock(configs, i)
            for i in range(configs.depth)])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seq_len = x_enc.shape[1]
        dec_out = 0

        for i, enc in enumerate(self.encoders):
            out_enc = enc(x_enc)
            dec_out += out_enc[:, -seq_len:, :]
            ar_roll = torch.zeros((x_enc.shape[0], self.configs.token_len, x_enc.shape[2])).to(x_enc.device)
            ar_roll = torch.cat([ar_roll, out_enc], dim=1)[:, :-self.configs.token_len, :]
            x_enc = x_enc - ar_roll

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

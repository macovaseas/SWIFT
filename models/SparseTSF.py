import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

wavelet_filter_lengths = {
    "haar": 2,
    "db1": 2,
    "db2": 4,
    "db3": 6,
    "db4": 8,
    "db5": 10,
    "db6": 12,
    "db7": 14,
    "db8": 16,
    "db9": 18,
    "db10": 20,
    "sym2": 4,
    "sym3": 6,
    "sym4": 8,
    "sym5": 10,
    "sym6": 12,
    "sym7": 14,
    "sym8": 16,
    "coif1": 6,
    "coif2": 12,
    "coif3": 18,
    "coif4": 24,
    "coif5": 30,
    "bior1.1": 2,  # 2, 2 (decomposition and reconstruction filters)
    "bior2.2": 6,  # 6, 2 (decomposition and reconstruction filters)
    "bior3.3": 10,  # 10, 2 (decomposition and reconstruction filters)
    "bior4.4": 14,  # 14, 2 (decomposition and reconstruction filters)
    "rbio1.1": 2,  # 2, 2 (reverse biorthogonal)
    "rbio2.2": 6,  # 6, 2 (reverse biorthogonal)
    "rbio3.3": 10,  # 10, 2 (reverse biorthogonal)
    "rbio4.4": 14,  # 14, 2 (reverse biorthogonal)
    "dmey": 102,  # Discrete Meyer wavelet
    "gaus1": 2,
    "gaus2": 4,
    "gaus3": 6,
    "mexh": "N/A",  # Mexican Hat wavelet, depends on scale
    "morl": "N/A"  # Morlet wavelet, depends on scale
}


def compute_dwt_dimensions(T, J, wav):
    filter_length = wavelet_filter_lengths[wav]
    P = filter_length - 1
    yh_lengths = []
    for j in range(1, J + 1):
        T = math.floor((T + P) / 2)
        yh_lengths.append(T)
    yl_length = T
    return yl_length, yh_lengths


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.individual = True #configs.individual

        self.wav = 'haar'
        self.J = 1

        self.dwt = DWT1DForward(wave=self.wav, J=self.J)
        self.idwt = DWT1DInverse(wave=self.wav)

        yl, _ = compute_dwt_dimensions(self.seq_len, self.J, self.wav)
        yl_, _ = compute_dwt_dimensions(self.pred_len, self.J, self.wav)

        if self.individual:
            self.projection = nn.ModuleList()
            for i in range(self.enc_in):
                self.projection.append(nn.Linear(yl, yl_))
        else:
            self.projection = nn.Linear(yl, yl_)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # normalization and permute     b,t,n -> b,n,t
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # DWT
        yl, yh = self.dwt(x)
        yh = yh[0]
        y = torch.stack([yl, yh], dim=-2)

        # Up sample
        if self.individual:
            y_ = torch.zeros([y.size(0), y.size(1), y.size(2), self.pred_len//2], dtype=y.dtype).to(y.device)
            for i in range(self.enc_in):
                y_[:, i, :, :] = self.dropout(self.projection[i](y[:, i, :, :]))
        else:
            y_ = self.dropout(self.projection(y))
        yl_, yh_ = y_[:, :, 0, :], [y_[:, :, 1, :]]

        # IDWT
        y_ = self.idwt((yl_, yh_))

        # permute and denorm
        y = y_.permute(0, 2, 1) + seq_mean

        return y

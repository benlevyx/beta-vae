"""implementing a beta-VAE from https://openreview.net/forum?id=Sy2fzU9gl

Borrowing code from https://github.com/AntixK/PyTorch-VAE
and https://github.com/sksq96/pytorch-vae
"""
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """beta-VAE CNN implementation
    """
    def __init__(self,
                 input_size: tuple,
                 encoder_layers: int = 4,
                 decoder_layers: int = 4,
                 latent_dim: int = 64,
                 beta: int = 4,
                 dropout: float = 0.1
                 ):
        """initialize a beta-VAE

        Args:
            input_size (tuple): (height, width, num_channels)
            encoder_layers (int, optional): Number of encoder layers. Defaults to 3.
            decoder_layers (int, optional): Number of decoder layers. Defaults to 3.
            latent_dim (int, optional): Size of latent dimension. Defaults to 64.
            beta (int, optional): Parameter for beta-VAE. Defaults to 4.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = FFNNVAEEncoder(input_size, latent_dim, encoder_layers, dropout)
        self.decoder = FFNNVAEDecoder(latent_dim, input_size, decoder_layers, dropout)
    
    def forward(self, input):
        x, input, mu, log_sigma2 = self.forward_step(input)
        loss = self.compute_loss(x, input, mu, log_sigma2)
        return x, loss
        
    def forward_step(self, input):
        mu, log_sigma2 = self.encoder(input)
        z = self.reparameterize(mu, log_sigma2)
        x = self.decoder(z)
        return x, input, mu, log_sigma2
    
    def reparameterize(self, mu, log_sigma2):
        z = torch.randn_like(mu)
        sigma = torch.exp(log_sigma2 / 2)
        return z * sigma + mu
    
    def compute_loss(self, pred, trg, mu, log_sigma2):
        """From https://github.com/matthew-liu/beta-vae/blob/master/models.py
        """
        # reconstruction losses are summed over all elements and batch
        reconstruction = F.binary_cross_entropy(pred, trg, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        kld = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())

        return (reconstruction + self.beta * kld) / trg.shape[0]  # divide total loss by batch size

    def embed(self, input):
        return self.reparameterize(*self.encoder(input))
    
    def decode(self, z):
        return self.decoder(z)
    

# ---------- FFNN Encoder and Decoder ---------- #
class FFNNVAEEncoder(nn.Module):
    """beta-VAE FFNN encoder
    """
    def __init__(self,
                 input_size,
                 latent_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.1
                 ):
        super().__init__()
        
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.flatten = nn.Flatten()

        layers = []
        # Building layers in reverse        
        d_in = self.latent_dim * 2
        d_out = int(d_in * 2)

        layers = []
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(d_out, d_in),
                nn.Dropout(self.dropout),
                nn.ReLU()
            ])
            d_in = d_out
            d_out = int(d_out * 2)
        layers = layers[::-1]
        
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_log_sigma2 = nn.Linear(latent_dim * 2, latent_dim)

        self.fc_in = nn.Linear(input_size[0] * input_size[1] * input_size[2], d_in)
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, input):
        z = self.fc_in(self.flatten(input))
        z = self.encoder(z)
        mu = self.fc_mu(z)
        log_sigma2 = self.fc_log_sigma2(z)
        return mu, log_sigma2
        
        
class FFNNVAEDecoder(nn.Module):
    """beta-VAE FFNN decoder

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 latent_dim,
                 output_size: tuple,
                 num_layers: int = 4,
                 dropout: float = 0.1
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Building layers from latent onwards
        d_in = self.latent_dim
        d_out = int(d_in * 2)
        
        layers = []
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(d_in, d_out),
                nn.Dropout(self.dropout),
                nn.ReLU()
            ])
            d_in = d_out
            d_out = int(d_out * 2)
        layers = layers

        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_in, output_size[0] * output_size[1] * output_size[2]),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        x = self.decoder(input)
        x = self.fc_out(x)
        return x.view(-1, *self.output_size)


# ---------- CNN Encoder and Decoder ---------- #

class CNNBetaVAE(nn.Module):
    """beta-VAE CNN implementation
    """
    def __init__(self,
                 input_size: tuple,
                 encoder_layers: int = 4,
                 decoder_layers: int = 4,
                 kernel_size: int = 3,
                 latent_dim: int = 64,
                 beta: int = 4
                 ):
        """initialize a beta-VAE
        Args:
            input_size (tuple): (height, width, num_channels)
            encoder_layers (int, optional): Number of encoder layers. Defaults to 3.
            decoder_layers (int, optional): Number of decoder layers. Defaults to 3.
            latent_dim (int, optional): Size of latent dimension. Defaults to 64.
            beta (int, optional): Parameter for beta-VAE. Defaults to 4.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = CNNBetaVAEEncoder(input_size, latent_dim, encoder_layers, kernel_size)
        self.decoder = CNNBetaVAEDecoder(latent_dim, input_size, decoder_layers, kernel_size)
        
    def forward(self, input):
        x, input, mu, log_sigma2 = self.forward_step(input)
        loss = self.compute_loss(x, input, mu, log_sigma2)
        return x, loss
        
    def forward_step(self, input):
        mu, log_sigma2 = self.encoder(input)
        z = self.reparameterize(mu, log_sigma2)
        x = self.decoder(z)
        return x, input, mu, log_sigma2
    
    def reparameterize(self, mu, log_sigma2):
        z = torch.randn_like(mu)
        sigma = torch.exp(log_sigma2 / 2)
        return z * sigma + mu
    
    def compute_loss(self, pred, trg, mu, log_sigma2):
        """From https://github.com/matthew-liu/beta-vae/blob/master/models.py
        """
        # reconstruction losses are summed over all elements and batch
        reconstruction = F.binary_cross_entropy(pred, trg, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        kld = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())

        return (reconstruction + self.beta * kld) / trg.shape[0]  # divide total loss by batch size    
    

class CNNBetaVAEEncoder(nn.Module):
    """beta-VAE CNN encoder
    """
    def __init__(self,
                 input_size,
                 latent_dim: int = 64,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 num_filters: int = 32,
                 stride: int = 2,
                 ):
        super().__init__()
        
        self.img_size = input_size[1:]
        self.num_channels = input_size[0]
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride

        d_in, d_out = self.num_channels, num_filters
        layers = []
        for _ in range(self.num_layers):
            layers.extend([
                nn.Conv2d(d_in, d_out, kernel_size=self.kernel_size, stride=self.stride, padding=1),
                nn.BatchNorm2d(d_out),
                nn.LeakyReLU()
            ])
            d_in = d_out
            d_out = d_out * 2
        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)
        
        # TODO: Figure out how large to make the bridge
        encoded_size = d_out * 2
        self.fc_mu = nn.Linear(encoded_size, latent_dim)
        self.fc_log_sigma2 = nn.Linear(encoded_size, latent_dim)
    
    def forward(self, input):
        z = self.encoder(input)
        mu = self.fc_mu(z)
        log_sigma2 = self.fc_log_sigma2(z)
        return mu, log_sigma2
        
        
class CNNBetaVAEDecoder(nn.Module):
    """beta-VAE CNN decoder

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 latent_dim,
                 output_size: tuple,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 num_filters: int = 128,
                 stride: int = 2
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        
        d_in, d_out = num_filters, int(num_filters / 2)
        self.bridge = nn.Linear(latent_dim, d_in * 4 * 4)
        
        layers = []
        for _ in range(self.num_layers):
            layers.extend([
                nn.ConvTranspose2d(d_in, d_out,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(d_out),
                nn.LeakyReLU()
            ])
            d_in = d_out
            d_out = int(d_out / 2)

        self.decoder = nn.Sequential(*layers)
        
        self.bridge_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in * 4 * 32 * 32, self.output_size[0] * self.output_size[1] * self.output_size[2])
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.output_size[0], self.output_size[0],
                      self.kernel_size, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        x = self.bridge(input).view(-1, self.num_filters, 4, 4)
        x = self.decoder(x)
        x = self.bridge_out(x)
        x = x.view(-1, *self.output_size)
        return self.conv_out(x)

"""implementing a beta-VAE from https://openreview.net/forum?id=Sy2fzU9gl

Borrowing code from https://github.com/AntixK/PyTorch-VAE
and https://github.com/sksq96/pytorch-vae
"""
from typing import Union

import torch
from torch import nn


class BetaVAE(nn.Module):
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
        self.encoder = BetaVAEEncoder(input_size, latent_dim, encoder_layers, kernel_size)
        self.decoder = BetaVAEDecoder(latent_dim, input_size, decoder_layers, kernel_size)
        
    def forward(self, input):
        mu, log_sigma2 = self.encoder(input)
        z = self.reparameterize(mu, log_sigma2)
        x = self.decoder(z)
        return z, input, mu, log_sigma2
    
    def reparameterize(self, mu, log_sigma2):
        z = torch.randn_like(mu)
        sigma = torch.exp(log_sigma2 / 2)
        return z * sigma + mu
    

class BetaVAEEncoder(nn.Module):
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
        
        self.img_size = input_size[:2]
        self.num_channels = input_size[2]
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
        encoded_size = d_out * 4
        self.fc_mu = nn.Linear(encoded_size, latent_dim)
        self.fc_log_sigma2 = nn.Linear(encoded_size, latent_dim)
    
    def forward(self, input):
        z = self.encoder(input)
        mu = self.fc_mu(z)
        log_sigma2 = self.fc_log_sigma2(z)
        return mu, log_sigma2
        
        
class BetaVAEDecoder(nn.Module):
    """beta-VAE CNN decoder

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 latent_dim,
                 output_size: tuple,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 num_filters: int = 256,
                 stride: int = 2
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        
        d_in, d_out = num_filters, num_filters / 2
        self.bridge = nn.Linear(latent_dim, d_in * 4)
        
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
            d_out = d_out / 2

        self.decoder = nn.Sequential(*layers)
        
        self.final_out = nn.Sequential(
            nn.ConvTranspose2d(d_out, d_out,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU(),
            nn.Conv2d(d_out, self.output_size[2],
                      kernel_size=self.kernel_size,
                      padding=1),
            nn.Tanh()
        )
    
    def forward(self, input):
        x = self.bridge(input).view(-1, self.kernel_size, 2, 2)
        x = self.decoder(x)
        return self.final_out(x)

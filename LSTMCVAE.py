import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class LSTMCVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    '''input: [batch_size,var] condition:[batch_size,condition_length,var]'''

    def __init__(self, input_size, latent_size, identity=0):
        super(LSTMCVAE, self).__init__()

        self.identity = identity
        encoder_size = [10,20,10]
        decoder_size = np.array([10,20,10])
        lstm_size=np.array([10,20,10])
        condition_length=latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_size[0]),
            # nn.BatchNorm1d(encoder_size[0]),
            nn.Tanh(),
            nn.Linear(encoder_size[0], encoder_size[1]),
            # nn.BatchNorm1d(encoder_size[1]),
            nn.Tanh(),
            nn.Linear(encoder_size[1], encoder_size[2]),
            # nn.BatchNorm1d(encoder_size[2]),
            nn.Tanh(),
        )
        self.encoder_mean = nn.Linear(encoder_size[2], latent_size)
        self.encoder_log_std = nn.Linear(encoder_size[2], latent_size)

        self.condition_generator=nn.LSTM(input_size,condition_length,num_layers=3,batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_length, decoder_size[0]),
            # nn.BatchNorm1d(decoder_size[0]),
            nn.Tanh(),
            nn.Linear(decoder_size[0], decoder_size[1]),
            # nn.BatchNorm1d(decoder_size[1]),
            nn.Tanh(),
            nn.Linear(decoder_size[1], decoder_size[2]),
            # nn.BatchNorm1d(decoder_size[2]),
            nn.Tanh(),
            nn.Linear(decoder_size[2], input_size),
        )

    def encode(self, x):
        # print("\033[0;34mencoder\033[0m", x.shape, x.dtype)
        h1 = self.encoder(x)
        mean = self.encoder_mean(h1)
        log_std = self.encoder_log_std(h1)
        return mean, log_std

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, condition):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        # print('\033[0;33mz shape, condition shape\033[0m', z.shape, condition.shape,
        #       torch.flatten(condition, start_dim=1).shape)
        # print('condition shape',condition.shape)
        _,(con,__)=self.condition_generator(condition)
        con=F.relu(con)[0]
        # print(con.shape)
        decoder_input = torch.cat((z, con), dim=1)
        # print(decoder_input.shape)
        # print('\033[0;33mdecoder input size\033[0m', decoder_input.shape)
        recon = self.decode(decoder_input)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std, kl_weight=1.):
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_weight * kl_loss
        return loss

if __name__ == '__main__':
    input_size=10
    latent_size=5
    lstmcvae=LSTMCVAE(input_size,latent_size)
    input=torch.arange(20,dtype=torch.float).reshape(2,10)
    condition=torch.arange(400,dtype=torch.float).reshape(2,20,10)
    output=lstmcvae(input,condition)
    print(output)
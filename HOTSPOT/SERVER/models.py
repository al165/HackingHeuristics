import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, feature_size, hidden_size, out_size, dropout=0.2):
        super(SimpleLinear, self).__init__()
        self.model_type = "Linear"

        # Encoders:
        sizes = [feature_size]
        if type(hidden_size) == int:
            sizes.append(hidden_size)
        else:
            try:
                sizes.extend(hidden_size)
            except TypeError:
                raise TypeError("`hidden_size` must be int or iterable")

        sizes.append(out_size)

        self.encoders = nn.ModuleList()
        for i, in_f in enumerate(sizes[:-1]):
            out_f = sizes[i + 1]
            self.encoders.append(nn.Linear(in_features=in_f, out_features=out_f))

    def forward(self, x):
        for enc in self.encoders[:-1]:
            x = F.relu(enc(x))
            x = self.dropout(x)

        x = F.relu(enc(x))


class SimpleNetwork(nn.Module):
    def __init__(
        self,
        sizes,
        dropout=0.4,
        batchnorm=False,
        batchnorm_kwargs=dict(),
        nonlinearity=nn.ReLU,
        nonlinearity_kwargs=dict(),
        final_layer=nn.Sigmoid,
        final_layer_kwargs=dict(),
    ):
        super(SimpleNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.sizes = sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers.append(
                nn.Linear(sizes[i], sizes[i + 1], bias=not batchnorm),
            )

            if i < len(sizes) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1], **batchnorm_kwargs))
                else:
                    layers.append(nn.Dropout(dropout))

                layers.append(nonlinearity(**nonlinearity_kwargs))

        layers.append(final_layer(**final_layer_kwargs))

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        output = self.stack(x)
        return output

    def get_device(self):
        return next(self.parameters()).device

    def init_latent(self, n=1):
        return torch.randn(n, self.sizes[0]).to(self.get_device())


class LinearAE(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.2):
        super(LinearAE, self).__init__()

        self.model_type = "AE"

        self._latent_size = latent_size

        # Encoder:
        self.enc1 = nn.Linear(in_features=feature_size, out_features=hidden_size)
        self.enc2 = nn.Linear(in_features=hidden_size, out_features=latent_size)

        # Decoder:
        self.dec1 = nn.Linear(in_features=latent_size, out_features=hidden_size)
        self.dec2 = nn.Linear(in_features=hidden_size, out_features=feature_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        z = self.encode(x)
        R = self.decode(z)

        return R, z

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = self.dropout(x)
        z = self.enc2(x)  # .view(-1, self._latent_size)
        return z

    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = self.dropout(x)
        R = self.dec2(x)  # .unsqueeze(1)
        return R


class LinearVAE(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.2):
        super(LinearVAE, self).__init__()

        self.model_type = "VAE"

        self._latent_size = latent_size

        # Encoders:
        sizes = [feature_size]
        if type(hidden_size) == int:
            sizes.append(hidden_size)
        else:
            try:
                sizes.extend(hidden_size)
            except TypeError:
                raise TypeError("`hidden_size` must be int or iterable")

        self.encoders = nn.ModuleList()
        for i, in_f in enumerate(sizes[:-1]):
            out_f = sizes[i + 1]
            self.encoders.append(nn.Linear(in_features=in_f, out_features=out_f))

        self.enc_mu = nn.Linear(in_features=sizes[-1], out_features=latent_size)
        self.enc_log_var = nn.Linear(in_features=sizes[-1], out_features=latent_size)

        # Decoder:
        self.decoders = nn.ModuleList()
        sizes.append(latent_size)
        sizes = list(reversed(sizes))
        for i, in_f in enumerate(sizes[:-1]):
            out_f = sizes[i + 1]
            self.decoders.append(nn.Linear(in_features=in_f, out_features=out_f))

        self.dropout = nn.Dropout(p=dropout)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        mu = self.encode(x)
        log_var = self.enc_log_var(x)
        sigma = torch.exp(log_var)

        z = mu + sigma * self.N.sample(mu.shape)

        self.kl = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - sigma, dim=1), dim=0
        )

        y = self.decode(z)

        return y, z

    def encode(self, x):
        """Deterministic encoding into latent space."""

        for enc in self.encoders:
            x = F.relu(enc(x))
            x = self.dropout(x)

        mu = self.enc_mu(x)
        mu = torch.tanh(mu)

        return mu

    def decode(self, z):
        y = z
        for dec in self.decoders[:-1]:
            y = F.relu(dec(y))
            y = self.dropout(y)

        y = self.decoders[-1](y)

        return y

    def to(self, device):
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        return super().to(device)


class SampleRNN(nn.Module):
    def __init__(self, pos_size, embedding_size, latent_size, hidden_size):
        super(SampleRNN, self).__init__()

        self._hidden_size = hidden_size

        self.enc1 = nn.Linear(pos_size + embedding_size + hidden_size, latent_size)
        self.enc2 = nn.Linear(latent_size, hidden_size)

        self.dec1 = nn.Linear(pos_size + hidden_size, latent_size)
        self.dec2 = nn.Linear(latent_size, embedding_size)

    def forward(self, pos, e_s, h):
        # encode current sequence
        z = self.encode(pos, e_s, h)

        # decode current sequence
        y = self.decode(pos, z)

        return y, z

    def encode(self, pos, e_s, h):
        x = torch.cat([pos, e_s, h], 1)
        z = self.enc1(x)
        z = F.relu(z)
        z = self.enc2(z)

        return z

    def decode(self, pos, z):
        y = torch.cat([pos, z], 1)
        y = self.dec1(y)
        y = F.relu(y)
        y = self.dec2(y)

        return y

    def encode_sequence(self, pos, e_s, z=None):
        """
        `pos.shape = (num_samples, grid_size)`
        `e_s.shape = (num_samples, embedding_size)`
        `z.shape   = (1, hidden_size)`
        """

        if z is None:
            z = self.init_hidden()

        for i in range(len(pos)):
            z = self.encode(pos[i : i + 1], e_s[i : i + 1], z)

        return z

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size).to(self.get_device())

    def get_device(self):
        return next(self.parameters()).device


class ExtendedRNN(nn.Module):
    def __init__(self, cat_size, input_size, hidden_size, output_size, dropout=0.1):
        """
        Adapted from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
        Similar to SampleRNN, but inclusion of category (embedding) input.
        """
        super(ExtendedRNN, self).__init__()

        self._hidden_size = hidden_size

        self.i2h = nn.Linear(cat_size + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(cat_size + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size).to(self.get_device())

    def get_device(self):
        return next(self.parameters()).device

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn


class SNPAD(nn.Module):
    def __init__(self, data_dim):
        super(SNPAD, self).__init__()
        self.N = 1
        self.encode = nn.Sequential(
            nn.Linear(in_features=data_dim[0], out_features=600, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=600, out_features=200, bias=True)
        )
        self.decode = nn.Sequential(
            nn.Linear(in_features=(data_dim[2]), out_features=200, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=200, out_features=1, bias=True)
        )
        self.R = nn.Sequential(
            nn.Linear(in_features=data_dim[0]+1, out_features=data_dim[1], bias=True),
            nn.Tanh(),
            nn.Linear(in_features=data_dim[1], out_features=data_dim[2], bias=True),
        )
        self.functions = nn.Sequential(
            nn.Linear(in_features=data_dim[2], out_features=data_dim[1], bias=True),
            nn.Tanh(),
            nn.Linear(in_features=data_dim[1], out_features=data_dim[2] * 2, bias=True),
        )
        self.predict = nn.Sequential(
            nn.Linear(in_features=(data_dim[2] * 2), out_features=600, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=600, out_features=200, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=5, bias=True),
        )
        self.alpha = 0.2
        self.beta = 0.1

    def generateZ(self, data):
        latent = self.encode(data)
        mu = latent[:, 0:self.data_dim[2]]
        logvar = latent[:, self.data_dim[2]:self.data_dim[2] * 2]
        z = self.reparameterize(mu, logvar)
        return z

    def generateFunctions(self, O):
        r = self.R(O)
        r = torch.mean(r, dim=0)
        self.r = r
        functions = self.functions(r)
        mu = functions[0:200]
        logvar = functions[200:400]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.N, 200).cuda()
        functions = mu + eps * std
        return functions, mu, logvar

    def forward(self, unlabel_data, label_data, label, O, isTraining):

        if isTraining:
            z_labeled = self.encode(label_data)
            z_unlabeled = self.encode(unlabel_data)
            shape = z_labeled.shape
            functions, mu, logvar  = self.generateFunctions(O)
            loss_KL = self.KL_loss(mu, logvar)
            function = functions.repeat(shape[0], 1)
            z_labeled = z_labeled.repeat(1, self.N).view(-1, shape[1])
            z_final = torch.cat((function, z_labeled), dim=1)
            labels = label.repeat(1, self.N).view(-1, 1)
            predictions = self.predict(z_final)
            loss_prediction = self.loss_l1(predictions, labels)
            shape = z_unlabeled.shape
            function = functions.repeat(shape[0], 1)
            z_unlabeled = z_unlabeled.repeat(1, self.N).view(-1, shape[1])
            z_final_unlabeled = torch.cat((function, z_unlabeled), dim=1)
            predictions = self.predict(z_final_unlabeled)
            loss_u = self.loss_u1(predictions)

            loss = loss_prediction + self.alpha * loss_u + self.beta * loss_KL
            return loss
        else:
            z_unlabeled = self.encode(unlabel_data)
            shape = z_unlabeled.shape
            functions, mu, logvar = self.generateFunctions(O)
            functions = functions.repeat(shape[0], 1)
            z = z_unlabeled.repeat(1, self.N).view(-1, shape[1])
            z_final = torch.cat((functions, z), dim=1)
            predictions = self.predict(z_final)
            predictions = torch.sum(predictions ** 2, dim=1)
            predictions = predictions.view(-1, self.N)

            return predictions

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_l1(self, prediction, label):
        dist_l = prediction ** 2
        dist_l = torch.unsqueeze(torch.sum(dist_l, dim=1), dim=1)
        dist_l = torch.where(label == 0, dist_l, dist_l ** (-1))
        loss_l = torch.sum(dist_l, dim=0)
        return loss_l

    def loss_u1(self, predictions):
        dist_u = predictions ** 2
        dist_u = torch.sum(dist_u, dim=1)
        loss_u = torch.sum(dist_u, dim=0)
        return loss_u

    def KL_loss(self, mu, logvar):
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0))
        return KLD


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Word2DM(nn.Module):

    def __init__(self, vocab_size, n_dim, m_dim):
        super(Word2DM, self).__init__()

        init_range = 0.5 / (n_dim * m_dim)

        self.B_target = nn.Embedding(vocab_size, n_dim * m_dim, sparse=True)
        self.B_target.weight.data.uniform_(-init_range, init_range)

        self.B_context = nn.Embedding(vocab_size, n_dim * m_dim, sparse=True)
        self.B_context.weight.data.uniform_(-init_range, init_range)

        self.n_dim = n_dim
        self.m_dim = m_dim

    def forward(self, target_indices, context_indices, neg_indices, batch_size):

        B_targets = self.B_target(target_indices).view(-1, self.n_dim, self.m_dim)
        B_contexts = self.B_context(context_indices).view(-1, self.n_dim, self.m_dim)
        B_negs = self.B_context(neg_indices).view(-1, len(neg_indices[0]), self.n_dim, self.m_dim)

        # Efficient optimisation function (see 3rd last page in notebook)
        B_targets = torch.transpose(B_targets, dim0=1, dim1=2)
        C_contexts = torch.bmm(B_targets, B_contexts)
        context_sims = torch.sum(C_contexts**2, dim=(-1, -2))
        context_log_sigmoids = F.logsigmoid(context_sims).squeeze()
        C_negs = torch.matmul(B_targets.unsqueeze(dim=1), B_negs)
        neg_sims = torch.sum(C_negs**2, dim=(-1, -2))
        neg_log_sigmoids = F.logsigmoid(-neg_sims).squeeze()

        loss = - context_log_sigmoids - neg_log_sigmoids.sum(dim=-1)
        return torch.mean(loss)

    def get_density_matrices(self):
        B_target = self.B_target.weight.view(-1, self.n_dim, self.m_dim)
        dms = torch.matmul(B_target, torch.transpose(B_target, dim0=1, dim1=2)).detach()
        return dms

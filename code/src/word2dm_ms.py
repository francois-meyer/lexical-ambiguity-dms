import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Word2DMMS(nn.Module):

    def __init__(self, vocab_size, n_dim, m_dim, setup, update, sim_metric, init):
        super(Word2DMMS, self).__init__()

        self.setup = setup
        self.update = update
        self.sim_metric = sim_metric

        if init == 0:
            target_init_range = 0.5 / (n_dim * m_dim)
            context_init_range = 0.5 / (n_dim * m_dim)
        elif init == 1:
            target_init_range = 0
            context_init_range = 0.5 / (n_dim * m_dim)
        elif init == 2:
            target_init_range = 0
            context_init_range = 0

        self.B_target = nn.Embedding(vocab_size, n_dim * m_dim, sparse=True)
        self.B_target.weight.data.uniform_(-target_init_range, target_init_range)

        if setup == "many2many":
            self.B_context = nn.Embedding(vocab_size, n_dim * m_dim, sparse=True)
        elif setup == "many2one":
            self.B_context = nn.Embedding(vocab_size, n_dim, sparse=True)

        self.B_context.weight.data.uniform_(-context_init_range, context_init_range)

        self.n_dim = n_dim
        self.m_dim = m_dim

    def forward(self, target_indices, context_indices, neg_indices, batch_size):

        B_targets = self.B_target(target_indices).view(-1, self.n_dim, self.m_dim)

        if self.setup == "many2many":
            B_contexts = self.B_context(context_indices).view(-1, self.n_dim, self.m_dim)
            B_negs = self.B_context(neg_indices).view(-1, len(neg_indices[0]), self.n_dim, self.m_dim)
        elif self.setup == "many2one":
            B_contexts = self.B_context(context_indices).unsqueeze(2)
            B_negs = self.B_context(neg_indices).view(-1, len(neg_indices[0]), self.n_dim)

        if self.setup == "many2many" and self.update == "one":
            B_dots = B_targets * B_contexts
            B_dots = torch.sum(B_dots, dim=1)

            if self.sim_metric == "cosine":
                target_norms = torch.sqrt(torch.sum(B_targets**2, dim=1))
                context_norms = torch.sqrt(torch.sum(B_contexts**2, dim=1))
                B_dots = B_dots / (target_norms * context_norms)

            max_indices = torch.argmax(B_dots, dim=1)

            b_targets = B_targets[torch.arange(0, len(B_targets)), :, max_indices]
            b_contexts = B_contexts[torch.arange(0, len(B_contexts)), :, max_indices]
            b_negs = B_negs[torch.arange(0, len(B_negs)), :, :, max_indices]

            context_sims = torch.sum(b_targets * b_contexts, dim=-1)
            neg_sims = torch.bmm(b_negs, b_targets.unsqueeze(2)).squeeze()

        elif self.setup == "many2many" and self.update == "all":
            context_sims = torch.sum(B_targets * B_contexts, dim=(2, 1))
            neg_sims = torch.sum(B_targets.unsqueeze(1) * B_negs, dim=(-1, -2))

        elif self.setup == "many2one":
            B_dots = B_targets * B_contexts
            B_dots = torch.sum(B_dots, dim=1)
            if self.sim_metric == "cosine":
                target_norms = torch.sqrt(torch.sum(B_targets**2, dim=1))
                context_norms = torch.sqrt(torch.sum(B_contexts**2, dim=1))
                B_dots = B_dots / (target_norms * context_norms)
            max_indices = torch.argmax(B_dots, dim=1)
            b_targets = B_targets[torch.arange(0, len(B_targets)), :, max_indices]

            context_sims = torch.sum(b_targets * B_contexts.unsqueeze(-1), dim=-1)
            neg_sims = torch.bmm(B_negs, b_targets.unsqueeze(2)).squeeze()

        context_log_sigmoids = F.logsigmoid(context_sims).squeeze()
        neg_log_sigmoids = torch.sum(F.logsigmoid(-neg_sims), dim=1)

        loss = - context_log_sigmoids - neg_log_sigmoids
        return torch.mean(loss)

    def get_component_matrices(self):
        B_target = self.B_target.weight.view(-1, self.n_dim, self.m_dim).detach()
        return B_target

    def get_density_matrices(self):
        B_target = self.B_target.weight.view(-1, self.n_dim, self.m_dim)
        dms = torch.matmul(B_target, torch.transpose(B_target, dim0=1, dim1=2)).detach()
        return dms

"""
Calibrated One-class classifier for Unsupervised Time series Anomaly detection (COUTA)
@author: Hongzuo Xu (hongzuo.xu@gmail.com)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import RandomState
from torch.utils.data import DataLoader
from deepod.utils.utility import get_sub_seqs
from deepod.core.networks.ts_network_tcn import TcnResidualBlock
from deepod.core.base_model import BaseDeepAD


class COUTA(BaseDeepAD):
    """
    COUTA class for Calibrated One-class classifier for Unsupervised Time series Anomaly detection

    Parameters
    ----------
    seq_len: integer, default=100
        sliding window length
    stride: integer, default=1
        sliding window stride
    epochs: integer, default=40
        the number of training epochs
    batch_size: integer, default=64
        the size of mini-batches
    lr: float, default=1e-4
        learning rate
    ss_type: string, default='FULL'
        types of perturbation operation type, which can be 'FULL' (using all
        three anomaly types), 'point', 'contextual', or 'collective'.
    hidden_dims: integer or list of integer, default=16,
        the number of neural units in the hidden layer
    rep_dim: integer, default=16
        the dimensionality of the feature space
    rep_hidden: integer, default=16
        the number of neural units of the hidden layer
    pretext_hidden: integer, default=16
    kernel_size: integer, default=2
        the size of the convolutional kernel in TCN
    dropout: float, default=0
        the dropout rate
    bias: bool, default=True
        the bias term of the linear layer
    alpha: float, default=0.1
        the weight of the classification head of NAC
    neg_batch_ratio: float, default=0.2
        the ratio of generated native anomaly examples
    es: bool, default=False
        early stopping
    seed: integer, default=42
        random state seed
    device: string, default='cuda'
    """
    def __init__(self, seq_len=100, stride=1,
                 epochs=40, batch_size=64, lr=1e-4, ss_type='FULL',
                 hidden_dims=16, rep_dim=16, rep_hidden=16, pretext_hidden=16,
                 kernel_size=2, dropout=0.0, bias=True,
                 alpha=0.1, neg_batch_ratio=0.2, train_val_pc=0.25,
                 epoch_steps=-1, prt_steps=1, device='cuda',
                 verbose=2, random_state=42,
                 ):
        super(COUTA, self).__init__(
            model_name='COUTA', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.ss_type = ss_type

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.rep_dim = rep_dim
        self.bias = bias

        self.alpha = alpha
        self.neg_batch_size = int(neg_batch_ratio * self.batch_size)
        self.max_cut_ratio = 0.5

        self.train_val_pc = train_val_pc

        self.net = None
        self.c = None
        self.test_df = None
        self.test_labels = None
        self.n_features = -1

        return

    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples, )
            Not used in unsupervised methods, present for API consistency by convention.
            used in (semi-/weakly-) supervised methods

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.n_features = X.shape[1]
        sequences = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        sequences = sequences[RandomState(42).permutation(len(sequences))]
        if self.train_val_pc > 0:
            train_seqs = sequences[: -int(self.train_val_pc * len(sequences))]
            val_seqs = sequences[-int(self.train_val_pc * len(sequences)):]
        else:
            train_seqs = sequences
            val_seqs = None

        self.net = COUTANet(
            input_dim=self.n_features,
            hidden_dims=self.hidden_dims,
            n_output=self.rep_dim,
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            out_dim=1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            bias=self.bias,
            pretext=True,
            dup=True
        )
        self.net.to(self.device)

        self.set_c(train_seqs)
        self.net = self.train(self.net, train_seqs, val_seqs)

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return

    def decision_function(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        test_sub_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        test_dataset = SubseqData(test_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        representation_lst = []
        representation_lst2 = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                representation_lst.append(x_output[0])
                representation_lst2.append(x_output[1])

        reps = torch.cat(representation_lst)
        dis = torch.sum((reps - self.c) ** 2, dim=1).data.cpu().numpy()

        reps_dup = torch.cat(representation_lst2)
        dis2 = torch.sum((reps_dup - self.c) ** 2, dim=1).data.cpu().numpy()
        dis = dis + dis2

        dis_pad = np.hstack([np.zeros(X.shape[0] - dis.shape[0]), dis])
        return dis_pad

    def train(self, net, train_seqs, val_seqs=None):
        val_loader = DataLoader(dataset=SubseqData(val_seqs),
                                batch_size=self.batch_size,
                                drop_last=False, shuffle=False) if val_seqs is not None else None
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        criterion_oc_umc = DSVDDUncLoss(c=self.c, reduction='mean')
        criterion_mse = torch.nn.MSELoss(reduction='mean')

        y0 = -1 * torch.ones(self.batch_size).float().to(self.device)

        net.train()
        for i in range(self.epochs):
            train_loader = DataLoader(dataset=SubseqData(train_seqs),
                                      batch_size=self.batch_size,
                                      drop_last=True, pin_memory=True, shuffle=True)

            rng = RandomState(seed=self.random_state+i)
            epoch_seed = rng.randint(0, 1e+6, len(train_loader))
            loss_lst, loss_oc_lst, loss_ssl_lst, = [], [], []
            for ii, x0 in enumerate(train_loader):
                x0 = x0.float().to(self.device)

                x0_output = net(x0)

                rep_x0 = x0_output[0]
                rep_x0_dup = x0_output[1]
                loss_oc = criterion_oc_umc(rep_x0, rep_x0_dup)

                neg_cand_idx = RandomState(epoch_seed[ii]).randint(0, self.batch_size, self.neg_batch_size)
                x1, y1 = create_batch_neg(batch_seqs=x0[neg_cand_idx],
                                          max_cut_ratio=self.max_cut_ratio,
                                          seed=epoch_seed[ii],
                                          return_mul_label=False,
                                          ss_type=self.ss_type)
                x1, y1 = x1.to(self.device), y1.to(self.device)
                y = torch.hstack([y0, y1])

                x1_output = net(x1)
                pred_x1 = x1_output[-1]
                pred_x0 = x0_output[-1]

                out = torch.cat([pred_x0, pred_x1]).view(-1)

                loss_ssl = criterion_mse(out, y)

                loss = loss_oc + self.alpha * loss_ssl

                net.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)
                loss_oc_lst.append(loss_oc)
                # loss_ssl_lst.append(loss_ssl)

            epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
            epoch_loss_oc = torch.mean(torch.stack(loss_oc_lst)).data.cpu().item()
            # epoch_loss_ssl = torch.mean(torch.stack(loss_ssl_lst)).data.cpu().item()

            # validation phase
            val_loss = np.NAN
            if val_seqs is not None:
                val_loss = []
                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        x_out = net(x)
                        loss = criterion_oc_umc(x_out[0], x_out[1])
                        loss = torch.mean(loss)
                        val_loss.append(loss)
                val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            if (i+1) % self.prt_steps == 0:
                print(
                    f'|>>> epoch: {i+1:02}  |   loss: {epoch_loss:.6f}, '
                    f'loss_oc: {epoch_loss_oc:.6f}, '
                    f'val_loss: {val_loss:.6f}'
                )

        return net

    def set_c(self, seqs, eps=0.1):
        """Initializing the center for the hypersphere"""
        dataloader = DataLoader(dataset=SubseqData(seqs), batch_size=self.batch_size,
                                drop_last=True, pin_memory=True, shuffle=True)
        z_ = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                rep = x_output[0]
                z_.append(rep.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


def create_batch_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL'):
    rng = np.random.RandomState(seed=seed)

    batch_size, l, dim = batch_seqs.shape
    cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)
    n_cut_dim = rng.randint(1, dim+1, size=batch_size)
    cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

    if type(batch_seqs) == np.ndarray:
        batch_neg = batch_seqs.copy()
        neg_labels = np.zeros(batch_size, dtype=int)
    else:
        batch_neg = batch_seqs.clone()
        neg_labels = torch.LongTensor(batch_size)

    if ss_type != 'FULL':
        pool = rng.randint(1e+6, size=int(1e+4))
        if ss_type == 'collective':
            pool = [a % 6 == 0 or a % 6 == 1 for a in pool]
        elif ss_type == 'contextual':
            pool = [a % 6 == 2 or a % 6 == 3 for a in pool]
        elif ss_type == 'point':
            pool = [a % 6 == 4 or a % 6 == 5 for a in pool]
        flags = rng.choice(pool, size=batch_size, replace=False)
    else:
        flags = rng.randint(1e+5, size=batch_size)

    n_types = 6
    for ii in range(batch_size):
        flag = flags[ii]

        # collective anomalies
        if flag % n_types == 0:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
            neg_labels[ii] = 1

        elif flag % n_types == 1:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
            neg_labels[ii] = 1

        # contextual anomalies
        elif flag % n_types == 2:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
            neg_labels[ii] = 2

        elif flag % n_types == 3:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
            neg_labels[ii] = 2

        # point anomalies
        elif flag % n_types == 4:
            batch_neg[ii, -1, cut_dim[ii]] = 2
            neg_labels[ii] = 3

        elif flag % n_types == 5:
            batch_neg[ii, -1, cut_dim[ii]] = -2
            neg_labels[ii] = 3

    if return_mul_label:
        return batch_neg, neg_labels
    else:
        neg_labels = torch.ones(batch_size).long()
        return batch_neg, neg_labels


class COUTANet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=32, rep_hidden=32, pretext_hidden=16,
                 n_output=10, kernel_size=2, dropout=0.2, out_dim=2,
                 bias=True, dup=True, pretext=True):
        super(COUTANet, self).__init__()

        self.layers = []

        if type(hidden_dims) == int: hidden_dims = [hidden_dims]
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias)]
        self.network = torch.nn.Sequential(*self.layers)
        self.l1 = torch.nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)
        self.l2 = torch.nn.Linear(rep_hidden, n_output, bias=bias)
        self.act = torch.nn.LeakyReLU()

        self.dup = dup
        self.pretext = pretext

        if dup:
            self.l1_dup = torch.nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)

        if pretext:
            self.pretext_l1 = torch.nn.Linear(hidden_dims[-1], pretext_hidden, bias=bias)
            self.pretext_l2 = torch.nn.Linear(pretext_hidden, out_dim, bias=bias)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        rep = self.l2(self.act(self.l1(out)))

        # pretext head
        if self.pretext:
            score = self.pretext_l2(self.act(self.pretext_l1(out)))

            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup, score
            else:
                return rep, score

        else:
            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup
            else:
                return rep


class SubseqData(Dataset):
    def __init__(self, x, y=None, w1=None, w2=None):
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]


class DSVDDUncLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DSVDDUncLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, rep2):
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2

        loss = 0.5*torch.exp(torch.mul(-1, var)) * (dis1+dis2) + 0.5*var

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



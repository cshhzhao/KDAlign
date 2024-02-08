import random
import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD


class USAD(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, hidden_dims=100, rep_dim=128,
                 epochs=100, batch_size=128, lr=1e-3,
                 es=1, train_val_pc=0.2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(USAD, self).__init__(
            model_name='USAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_size = hidden_dims
        self.train_val_pc = train_val_pc
        self.es = es

        self.model = None
        self.w_size = None

        return

    def fit(self, X, y=None):
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        self.w_size = seqs.shape[1] * seqs.shape[2]
        z_size = seqs.shape[1] * self.hidden_size

        if self.train_val_pc > 0:
            train_seqs = seqs[: -int(self.train_val_pc * len(seqs))]
            val_seqs = seqs[-int(self.train_val_pc * len(seqs)):]
        else:
            train_seqs = seqs
            val_seqs = None

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_seqs).float().view(([train_seqs.shape[0], self.w_size]))),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        if val_seqs is not None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_seqs).float().view(([val_seqs.shape[0], self.w_size]))),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            val_loader = None

        self.model = UsadModel(self.w_size, z_size)
        self.model = self.model.to(self.device)

        self.training(train_loader, val_loader)

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return

    def decision_function(self, x, labels=None):
        seqs = get_sub_seqs(x, seq_len=self.seq_len, stride=1)

        test_loader = DataLoader(TensorDataset(
            torch.from_numpy(seqs).float().view(([seqs.shape[0], self.w_size]))
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        results = self.testing(test_loader)
        y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                 results[-1].flatten().detach().cpu().numpy()])

        score_pad = np.hstack([0 * np.ones(x.shape[0] - y_pred.shape[0]), y_pred])
        return score_pad

    def training(self, train_loader, val_loader, opt_func=torch.optim.Adam):
        # if self.es:
        #     early_stopper = EarlyStopping(patience=7, model_name=self.__class__.__name__, verbose=False)
        # else:
        #     early_stopper = None

        optimizer1 = opt_func(list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
                              lr=self.lr)
        optimizer2 = opt_func(list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
                              lr=self.lr)

        for i in range(self.epochs):
            t1 = time.time()

            train_loss1 = []
            train_loss2 = []
            for [batch] in train_loader:
                batch = batch.to(self.device)

                # Train AE1
                loss1, loss2 = self.model.training_step(batch, i+1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                # Train AE2
                loss1, loss2 = self.model.training_step(batch, i+1)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                train_loss1.append(loss1)
                train_loss2.append(loss2)

            train_loss1 = torch.stack(train_loss1).mean().item()
            train_loss2 = torch.stack(train_loss2).mean().item()

            total_loss = train_loss1 + train_loss2

            t = time.time() - t1

            val_loss1 = np.nan
            val_loss2 = np.nan
            if val_loader is not None:
                outputs = [self.model.validation_step(batch.to(self.device), i+1) for [batch] in val_loader]
                result = self.model.validation_epoch_end(outputs)
                val_loss1, val_loss2 = result['val_loss1'], result['val_loss2']

            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1:3d}, '
                      f'training loss: {total_loss:.6f}, time: {t:.1f}')

        return

    def testing(self, test_loader, alpha=.5, beta=.5):
        results = []
        for [batch] in test_loader:
            batch = batch.to(self.device)
            w1 = self.model.decoder1(self.model.encoder(batch))
            w2 = self.model.decoder2(self.model.encoder(w1))
            results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))
        return results

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


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}


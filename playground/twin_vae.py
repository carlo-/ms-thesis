import os
import pickle
import json
from typing import Callable
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from thesis_tools.dtw import dynamic_time_warping_c, dynamic_time_warping_py


def _fast_dtw_with_model(a_eps, b_eps, worker_i, queue, model):

    model.eval()
    dev = model.device

    a_res = []
    b_res = []
    t = None
    if worker_i == 0:
        t = tqdm(desc='Aligned episodes', total=len(a_eps))

    for a_ep, b_ep in zip(a_eps, b_eps):
        with torch.no_grad():
            a = torch.tensor(a_ep, dtype=torch.float32, device=dev)
            b = torch.tensor(b_ep, dtype=torch.float32, device=dev)
            n, m = a.shape[0], b.shape[0]
            res = torch.zeros((n, m), dtype=torch.float32, device=dev)
            for i in range(n):
                sim_loss = model.compute_elementwise_sim_loss(a[i].repeat(m, 1), b, reparameterize=False)
                res[i] = sim_loss.mean(dim=1)
            precomputed_distances = res.cpu().numpy()

        cost, path = dynamic_time_warping_c(precomputed_distances)
        a_res += list(a_ep[path[:, 0]])
        b_res += list(b_ep[path[:, 1]])

        if t is not None:
            t.update()

    if worker_i == 0:
        t.close()

    if queue is not None:
        queue.put((worker_i, (a_res, b_res)))
    else:
        return a_res, b_res


class AutoencoderBase(nn.Module):

    def __init__(self, input_size: int, **kwargs):
        super(AutoencoderBase, self).__init__()
        self.input_size = input_size

    def reset_parameters(self):
        def reset(m):
            if m != self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset)

    def encode(self, x, **kwargs):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def loss_function(self, *args, **kwargs):
        raise NotImplementedError


class VAE(AutoencoderBase):

    # From : https://github.com/pytorch/examples/blob/5df464c46cf321ed1cc3df1e670358d7f5ae1887/vae/main.py#L39
    def __init__(self, input_size: int, hidden_size: int, z_size: int, **kwargs):
        super(VAE, self).__init__(input_size, **kwargs)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(hidden_size, z_size)
        self.fc32 = nn.Linear(hidden_size, z_size)

        self.fc4 = nn.Linear(z_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, input_size)

    def _encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, reparameterize=True, **kwargs):
        mu, logvar = self._encode(x.view(-1, self.input_size))
        if reparameterize:
            # return z
            return self._reparameterize(mu, logvar)
        else:
            return mu

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self._encode(x.view(-1, self.input_size))
        z = self._reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    # From VAE example code
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, recon_x, mu, logvar):
        bce = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), reduction='sum')
        kdl = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.5
        tot = bce + kdl
        return tot, bce, kdl


class SimpleAutoencoder(AutoencoderBase):

    def __init__(self, input_size: int, hidden_size: int, z_size: int, use_kdl_loss: bool = False, **kwargs):
        super(SimpleAutoencoder, self).__init__(input_size, **kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, z_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )

        self._loss_fn = nn.MSELoss(reduction='sum')
        self.use_kdl_loss = use_kdl_loss

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def loss_function(self, x, recon_x, z=None, logvar=None):
        mse = self._loss_fn(recon_x, x.view(-1, self.input_size))
        kdl = np.zeros(1)
        if self.use_kdl_loss:
            # kdl = (z.pow(2) + logvar.exp() - 1 - logvar).sum() * 0.5
            kdl = ((z - 1.0)**2.0).sum()
            tot = mse + kdl
        else:
            tot = mse
        return tot, mse, kdl


class TwinVAE:

    def __init__(self, a_spec: dict, b_spec: dict, z_size: int, net_class=None,
                 normalize_losses=False, loss_weights=None):
        self.a_spec = dict(a_spec)
        self.b_spec = dict(b_spec)
        self.z_size = z_size

        net_class = net_class or VAE
        self.a_vae = net_class(**a_spec, z_size=z_size) # type: AutoencoderBase
        self.b_vae = net_class(**b_spec, z_size=z_size) # type: AutoencoderBase

        self.a_loss = self.a_vae.loss_function
        self.b_loss = self.b_vae.loss_function
        self.sim_loss = nn.MSELoss(reduction='sum')
        self.sim_loss_no_red = nn.MSELoss(reduction='none')
        self.normalize_losses = normalize_losses
        self.loss_weights = loss_weights
        self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    def save(self, file_path: str):
        assert file_path.endswith(".pt")
        torch.save(dict(
            a_spec=self.a_spec,
            b_spec=self.b_spec,
            z_size=self.z_size,
            a_vae_state=self.a_vae.state_dict(),
            b_vae_state=self.b_vae.state_dict(),
        ), file_path)

    @staticmethod
    def load(file_path: str, net_class):
        assert file_path.endswith(".pt")
        d = torch.load(file_path, map_location=lambda st, loc: st)
        model = TwinVAE(
            a_spec=d['a_spec'],
            b_spec=d['b_spec'],
            z_size=d['z_size'],
            net_class=net_class,
        )
        model.a_vae.load_state_dict(d['a_vae_state'])
        model.b_vae.load_state_dict(d['b_vae_state'])
        model.eval()
        return model

    def reset_parameters(self):
        self.a_vae.reset_parameters()
        self.b_vae.reset_parameters()

    def compute_elementwise_sim_loss(self, a_data, b_data, **encode_kwargs):
        a_z = self.a_vae.encode(a_data, **encode_kwargs)
        b_z = self.b_vae.encode(b_data, **encode_kwargs)
        return self.sim_loss_no_red(a_z, b_z)

    def compute_losses(self, a_data, b_data, fixed_logvar=None):

        if isinstance(self.a_vae, VAE):
            a_z, a_recon_x, a_mu, a_logvar = self.a_vae(a_data)
            a_loss, a_recon_l, a_kdl_l = self.a_loss(a_data, a_recon_x, a_mu, a_logvar)

            b_z, b_recon_x, b_mu, b_logvar = self.b_vae(b_data)
            b_loss, b_recon_l, b_kdl_l = self.b_loss(b_data, b_recon_x, b_mu, b_logvar)
        else:
            a_z, a_recon_x = self.a_vae(a_data)
            a_loss, a_recon_l, a_kdl_l = self.a_loss(a_data, a_recon_x, a_z, fixed_logvar)

            b_z, b_recon_x = self.b_vae(b_data)
            b_loss, b_recon_l, b_kdl_l = self.b_loss(b_data, b_recon_x, b_z, fixed_logvar)

        sim_loss = self.sim_loss(a_z, b_z)

        if self.normalize_losses:
            a_loss /= self.a_vae.input_size
            b_loss /= self.b_vae.input_size
            sim_loss /= self.z_size

        if self.loss_weights is not None:
            a_loss *= self.loss_weights['a']
            b_loss *= self.loss_weights['b']
            sim_loss *= self.loss_weights['sim']

        total_loss = a_loss + b_loss + sim_loss

        return total_loss, a_loss, b_loss, sim_loss, a_recon_l, a_kdl_l, b_recon_l, b_kdl_l

    def compute_obs_sim_loss(self, a_obs, b_obs):
        with torch.no_grad():
            a = torch.tensor(a_obs[None], dtype=torch.float32, device=self.device)
            b = torch.tensor(b_obs[None], dtype=torch.float32, device=self.device)
            sim_loss = self.compute_elementwise_sim_loss(a, b, reparameterize=False)[0]
        return sim_loss.cpu().numpy()

    def cross_decode_a_to_b(self, a_obs):
        with torch.no_grad():
            a = torch.tensor(a_obs[None], dtype=torch.float32, device=self.device)
            z = self.a_vae.encode(a, reparameterize=False)
            recon_b = self.b_vae.decode(z)[0]
        return recon_b.cpu().numpy()

    def cross_decode_b_to_a(self, b_obs):
        with torch.no_grad():
            b = torch.tensor(b_obs[None], dtype=torch.float32, device=self.device)
            z = self.b_vae.encode(b, reparameterize=False)
            recon_a = self.a_vae.decode(z)[0]
        return recon_a.cpu().numpy()

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.a_vae = self.a_vae.to(device)
        self.b_vae = self.b_vae.to(device)
        self._device = device

    def train(self):
        self.a_vae.train()
        self.b_vae.train()

    def eval(self):
        self.a_vae.eval()
        self.b_vae.eval()

    def parameters(self):
        return list(self.a_vae.parameters()) + list(self.b_vae.parameters())

    def clone(self):
        c = TwinVAE(self.a_spec, self.b_spec, self.z_size, self.a_vae.__class__)
        c.a_vae.load_state_dict(self.a_vae.state_dict())
        c.b_vae.load_state_dict(self.b_vae.state_dict())
        c.to(self.device)
        return c


def _training_step(model: TwinVAE, optimizer: optim.Optimizer, data_loader, device, epoch: int, log_interval=1,
                   writer: SummaryWriter = None, global_step: int = None, fixed_logvar=None):
    model.train()
    avg_tot_loss = 0.0
    avg_sim_loss = 0.0
    avg_a_loss = 0.0
    avg_b_loss = 0.0
    avg_a_kdl_loss = 0.0
    avg_b_kdl_loss = 0.0
    avg_a_recon_loss = 0.0
    avg_b_recon_loss = 0.0
    ds_size = len(data_loader.dataset)

    for batch_idx, (a_data, b_data) in enumerate(data_loader):

        b_size = len(a_data)
        a_data = a_data.to(device)
        b_data = b_data.to(device)

        optimizer.zero_grad()
        losses = model.compute_losses(a_data, b_data, fixed_logvar)
        total_loss, a_loss, b_loss, sim_loss, a_recon_l, a_kdl_l, b_recon_l, b_kdl_l = losses

        avg_tot_loss += total_loss.item()
        avg_a_loss += a_loss.item()
        avg_b_loss += b_loss.item()
        avg_sim_loss += sim_loss.item()
        avg_a_kdl_loss += a_kdl_l.item()
        avg_b_kdl_loss += b_kdl_l.item()
        avg_a_recon_loss += a_recon_l.item()
        avg_b_recon_loss += b_recon_l.item()

        total_loss.backward()
        optimizer.step(None)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (sim: {:.6f}, a: {:.6f}, b: {:.6f})'.format(
                epoch, batch_idx * b_size, ds_size, 100. * batch_idx / len(data_loader), total_loss.item() / b_size,
                sim_loss.item() / b_size, a_loss.item() / b_size, b_loss.item() / b_size)
            )

    avg_tot_loss /= ds_size
    avg_a_loss /= ds_size
    avg_b_loss /= ds_size
    avg_sim_loss /= ds_size
    avg_a_kdl_loss /= ds_size
    avg_b_kdl_loss /= ds_size
    avg_a_recon_loss /= ds_size
    avg_b_recon_loss /= ds_size
    print('====> Epoch: {} Average loss: {:.4f} (sim: {:.6f}, a: {:.6f}, b: {:.6f})'.format(
        epoch, avg_tot_loss, avg_sim_loss, avg_a_loss, avg_b_loss
    ))

    if writer is not None:
        writer.add_scalar(tag='training_loss/avg_total', scalar_value=avg_tot_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_a_loss', scalar_value=avg_a_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_b_loss', scalar_value=avg_b_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_sim_loss', scalar_value=avg_sim_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_a_kdl_loss', scalar_value=avg_a_kdl_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_b_kdl_loss', scalar_value=avg_b_kdl_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_a_recon_loss', scalar_value=avg_a_recon_loss, global_step=global_step)
        writer.add_scalar(tag='training_loss/avg_b_recon_loss', scalar_value=avg_b_recon_loss, global_step=global_step)


def _evaluation_step(model: TwinVAE, data_loader, device, writer: SummaryWriter = None, global_step: int = None,
                     fixed_logvar=None):
    model.eval()
    avg_tot_loss = 0.0
    avg_sim_loss = 0.0
    avg_a_loss = 0.0
    avg_b_loss = 0.0
    avg_a_kdl_loss = 0.0
    avg_b_kdl_loss = 0.0
    avg_a_recon_loss = 0.0
    avg_b_recon_loss = 0.0
    ds_size = len(data_loader.dataset)

    with torch.no_grad():
        for i, (a_data, b_data) in enumerate(data_loader):

            a_data = a_data.to(device)
            b_data = b_data.to(device)

            losses = model.compute_losses(a_data, b_data, fixed_logvar)
            total_loss, a_loss, b_loss, sim_loss, a_recon_l, a_kdl_l, b_recon_l, b_kdl_l = losses

            avg_tot_loss += total_loss.item()
            avg_a_loss += a_loss.item()
            avg_b_loss += b_loss.item()
            avg_sim_loss += sim_loss.item()
            avg_a_kdl_loss += a_kdl_l.item()
            avg_b_kdl_loss += b_kdl_l.item()
            avg_a_recon_loss += a_recon_l.item()
            avg_b_recon_loss += b_recon_l.item()

    avg_tot_loss /= ds_size
    avg_a_loss /= ds_size
    avg_b_loss /= ds_size
    avg_sim_loss /= ds_size
    avg_a_kdl_loss /= ds_size
    avg_b_kdl_loss /= ds_size
    avg_a_recon_loss /= ds_size
    avg_b_recon_loss /= ds_size
    print('====> Test average loss: {:.4f} (sim: {:.6f}, a: {:.6f}, b: {:.6f})'.format(
        avg_tot_loss, avg_sim_loss, avg_a_loss, avg_b_loss
    ))

    if writer is not None:
        writer.add_scalar(tag='test_loss/avg_total', scalar_value=avg_tot_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_a_loss', scalar_value=avg_a_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_b_loss', scalar_value=avg_b_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_sim_loss', scalar_value=avg_sim_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_a_kdl_loss', scalar_value=avg_a_kdl_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_b_kdl_loss', scalar_value=avg_b_kdl_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_a_recon_loss', scalar_value=avg_a_recon_loss, global_step=global_step)
        writer.add_scalar(tag='test_loss/avg_b_recon_loss', scalar_value=avg_b_recon_loss, global_step=global_step)


def _visual_evaluation_step(model: TwinVAE, dataset, device, writer: SummaryWriter = None, global_step: int = None,
                            cycle_i=0, *, a_get_ag, a_get_g, b_get_ag, b_get_g):

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.cm as cm

    model.eval()
    num_examples = 3
    np_random = np.random.RandomState(120)

    done_idx = []

    fig = Figure(figsize=(8, 4*num_examples), dpi=100)
    fig.suptitle(f'Cycle {cycle_i}')
    canvas = FigureCanvasAgg(fig)

    while len(done_idx) < num_examples:
        i = len(done_idx)
        ep_i = np_random.choice(len(dataset.a_episodes), size=1).item()
        if ep_i in done_idx:
            continue

        a_ep = deepcopy(dataset.a_episodes[ep_i])
        b_ep = deepcopy(dataset.b_episodes[ep_i])

        a_ep_s = dataset.a_scaler.inverse_transform(a_ep, copy=True)
        b_ep_s = dataset.b_scaler.inverse_transform(b_ep, copy=True)

        goal_d = np.linalg.norm(a_get_ag(a_ep_s) - a_get_g(a_ep_s), axis=1)
        if np.abs(goal_d[0] - goal_d[-1]) < 0.1:
            # uninteresting episode
            continue

        done_idx.append(ep_i)

        with torch.no_grad():
            a_ep_t = torch.tensor(a_ep, dtype=torch.float32, device=device)
            b_ep_t = torch.tensor(b_ep, dtype=torch.float32, device=device)
            n, m = a_ep_t.shape[0], b_ep_t.shape[0]
            res = torch.zeros((n, m), dtype=torch.float32, device=device)
            for j in range(n):
                # foo = np.linalg.norm(a_ep_s[j, 18:21] - a_ep_s[j, -3:])
                # bar = np.linalg.norm(b_ep_s[:, 63:66] - b_ep_s[0, -7:-4], axis=1)
                # res[j] = torch.tensor((foo - bar)**2.0, dtype=torch.float32, device=device)
                sim_loss = model.compute_elementwise_sim_loss(a_ep_t[j].repeat(m, 1), b_ep_t, reparameterize=False)
                res[j] = sim_loss.mean(dim=1)
            res = res.cpu().numpy()

        sim = np.diag(res)
        cost, path = dynamic_time_warping_c(res)
        a_dtw_s = a_ep_s[path[:, 0]]
        b_dtw_s = b_ep_s[path[:, 1]]
        sim_dtw = res[path[:, 0], path[:, 1]]

        a_ag = a_get_ag(a_ep_s)
        a_g = a_get_g(a_ep_s)
        b_ag = b_get_ag(b_ep_s)
        b_g = b_get_g(b_ep_s)

        ax_l = fig.add_subplot(num_examples, 2, i*2 + 1)
        ax_r = fig.add_subplot(num_examples, 2, i*2 + 2)

        a = np.linalg.norm(a_ag - a_g, axis=1)
        b = np.linalg.norm(b_ag - b_g, axis=1)
        diff = np.abs(a - b)
        ax_l.plot(a, label="A")
        ax_l.plot(b, label="B")
        ax_l.plot(-(sim / (sim.max() + 1e-5) * diff.max()), label="sim*")
        ax_l.plot(-diff, label="diff")
        ax_l.grid()
        ax_l.legend()

        a = np.linalg.norm(a_get_ag(a_dtw_s) - a_get_g(a_dtw_s), axis=1)
        b = np.linalg.norm(b_get_ag(b_dtw_s) - b_get_g(b_dtw_s), axis=1)
        diff = np.abs(a - b)
        ax_r.plot(a, label="A")
        ax_r.plot(b, label="B")
        ax_r.plot(-(sim_dtw / (sim_dtw.max() + 1e-5) * diff.max()), label="sim*")
        ax_r.plot(-diff, label="diff")
        ax_r.grid()
        ax_r.legend()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    x = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    writer.add_image(tag='visual_eval', img_tensor=x, global_step=global_step, dataformats='HWC')
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(x)
    # plt.show()


class TwinDataset(Dataset):

    def __init__(self, a_episodes: list, b_episodes: list, a_states=None, b_states=None, copy=True):
        super(TwinDataset, self).__init__()
        assert len(a_episodes) == len(b_episodes)
        assert len(a_episodes) > 0

        if copy and a_states is not None:
            self.a_states = deepcopy(a_states)
        else:
            self.a_states = a_states or []

        if copy and b_states is not None:
            self.b_states = deepcopy(b_states)
        else:
            self.b_states = b_states or []

        if copy:
            self.a_episodes = deepcopy(a_episodes)
            self.b_episodes = deepcopy(b_episodes)
        else:
            self.a_episodes = a_episodes
            self.b_episodes = b_episodes

        self.a_scaler = StandardScaler()
        self.b_scaler = StandardScaler()

    def __getitem__(self, idx):
        return self.a_states[idx].astype(np.float32), self.b_states[idx].astype(np.float32)

    def __len__(self):
        return len(self.a_states)

    def split(self, shuffle=True, test_size=0.25, train_size=None, seed=None, copy=False, keep_scalers=True):
        a_train, a_test, b_train, b_test = train_test_split(self.a_episodes, self.b_episodes, test_size=test_size,
                                                            train_size=train_size, shuffle=shuffle, random_state=seed)
        first = TwinDataset(a_train, b_train, copy=copy)
        second = TwinDataset(a_test, b_test, copy=copy)

        if keep_scalers:
            first.a_scaler = deepcopy(self.a_scaler)
            first.b_scaler = deepcopy(self.b_scaler)
            second.a_scaler = deepcopy(self.a_scaler)
            second.b_scaler = deepcopy(self.b_scaler)

        return first, second

    @staticmethod
    def merge(dataset1, dataset2):
        assert isinstance(dataset1, TwinDataset)
        assert isinstance(dataset2, TwinDataset)
        return TwinDataset(
            a_episodes=(dataset1.a_episodes + dataset2.a_episodes),
            b_episodes=(dataset1.b_episodes + dataset2.b_episodes),
            copy=True,
        )

    def save(self, file_path: str, with_aligned_states=False):
        obj = dict(a_episodes=self.a_episodes, b_episodes=self.b_episodes)
        if with_aligned_states:
            obj += dict(a_states=self.a_states, b_states=self.b_states)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return TwinDataset(**obj, copy=False)

    def normalize(self):
        self.a_scaler.fit(np.zeros((1, 1)))
        self.b_scaler.fit(np.zeros((1, 1)))
        for i in range(2):
            desc = 'Fitted episodes' if i == 0 else 'Normalized episodes'
            for a_ep, b_ep in tqdm(zip(self.a_episodes, self.b_episodes), desc=desc, total=len(self.a_episodes)):
                if i == 0:
                    self.a_scaler.partial_fit(a_ep)
                    self.b_scaler.partial_fit(b_ep)
                else:
                    self.a_scaler.transform(a_ep, copy=False)
                    self.b_scaler.transform(b_ep, copy=False)

    def _extract_states(self):
        # equivalent to self.realign(lambda x, y: 0.0)
        self.a_states = []
        self.b_states = []
        for a_ep, b_ep in zip(self.a_episodes, self.b_episodes):
            min_len = min(len(a_ep), len(b_ep))
            self.a_states += list(a_ep[:min_len])
            self.b_states += list(b_ep[:min_len])

    def preprocess(self):
        if len(self.a_states) == 0 or len(self.b_states) != len(self.a_states):
            self._extract_states()

    def realign(self, distance_fn: Callable = None, vec_distance_fn: Callable = None):
        assert distance_fn is None and vec_distance_fn is not None or \
               distance_fn is not None and vec_distance_fn is None
        self.a_states = []
        self.b_states = []
        for a_ep, b_ep in tqdm(zip(self.a_episodes, self.b_episodes),
                               desc='Aligned episodes', total=len(self.a_episodes)):
            if distance_fn is not None:
                cost, path = dynamic_time_warping_py(a_ep, b_ep, distance_fn)
            else:
                cost, path = dynamic_time_warping_c(vec_distance_fn(a_ep, b_ep))
            self.a_states += list(a_ep[path[:, 0]])
            self.b_states += list(b_ep[path[:, 1]])

    def realign_with_model(self, model: TwinVAE, workers=6):

        if workers == 1:
            a_res, b_res = _fast_dtw_with_model(self.a_episodes, self.b_episodes, 0, None, model)
            self.a_states = a_res
            self.b_states = b_res
            return

        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        processes = []
        for i in range(workers):

            epj = len(self.a_episodes) // workers + 1
            min_i = i * epj
            max_i = (i + 1) * epj
            a_eps = self.a_episodes[min_i:max_i]
            b_eps = self.b_episodes[min_i:max_i]

            if i == 0:
                child_model = model
            else:
                child_model = model.clone()

            p = mp.Process(target=_fast_dtw_with_model, args=(a_eps, b_eps, i, queue, child_model))
            p.start()
            processes.append(p)

        a_states = [None] * workers
        b_states = [None] * workers
        for _ in range(workers):
            worker_i, (a_res, b_res) = queue.get()
            a_states[worker_i] = a_res
            b_states[worker_i] = b_res
            if worker_i == 0:
                print(f'Waiting for other workers...')

        self.a_states = sum(a_states, [])
        self.b_states = sum(b_states, [])

        for p in processes:
            p.join(timeout=5.0)

    @property
    def a_item_size(self):
        return self.a_episodes[0].shape[1]

    @property
    def b_item_size(self):
        return self.b_episodes[0].shape[1]


def train(*, training_set: TwinDataset, test_set: TwinDataset, local_dir: str, seed=42, learning_rate=1e-4,
          n_workers=1, n_cycles=50, n_epochs=3, batch_size=128, net_class=None, new_params_each_cycle=False,
          ae_kwargs=None, z_logvar=0.0, z_size=15, loss_weights=None, goal_extractors=None,
          init_with_goal_based_alignment=False):

    checkpoints_dir = local_dir + '/checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.manual_seed(seed)

    writer = SummaryWriter(log_dir=local_dir)

    ae_kwargs = ae_kwargs or dict()
    goal_extractors = goal_extractors or dict()

    model_config = dict(
        a_spec=dict(input_size=training_set.a_item_size, hidden_size=60, **ae_kwargs),
        b_spec=dict(input_size=training_set.b_item_size, hidden_size=60, **ae_kwargs),
        z_size=z_size,
        net_class=net_class,
        normalize_losses=False,
        loss_weights=loss_weights,
    )

    other_info = dict(
        learning_rate=learning_rate,
        new_params_each_cycle=new_params_each_cycle,
        n_cycles=n_cycles,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        training_set_size=len(training_set),
        test_set_size=len(test_set),
    )

    pp_info = json.dumps({**model_config, **other_info}, default=lambda x: x.__name__, sort_keys=True, indent=4)
    writer.add_text(text_string=pp_info, tag="config")
    model = TwinVAE(**model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_set.preprocess()
    test_set.preprocess()

    if init_with_goal_based_alignment:

        a_get_ag = goal_extractors['a_get_ag']
        a_get_g = goal_extractors['a_get_g']
        b_get_ag = goal_extractors['b_get_ag']
        b_get_g = goal_extractors['b_get_g']

        from scipy.spatial.distance import cdist

        def vec_dist(a_ep, b_ep):
            a_ep = training_set.a_scaler.inverse_transform(a_ep, copy=True)
            b_ep = training_set.b_scaler.inverse_transform(b_ep, copy=True)
            a = np.linalg.norm(a_get_ag(a_ep) - a_get_g(a_ep), axis=1)[..., None]
            b = np.linalg.norm(b_get_ag(b_ep) - b_get_g(b_ep), axis=1)[..., None]
            res = cdist(a, b).astype(np.float32)
            return res

        training_set.realign(vec_distance_fn=vec_dist)
        test_set.realign(vec_distance_fn=vec_dist)

    training_data_loader = DataLoader(dataset=training_set, num_workers=n_workers, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    fixed_logvar = None
    if z_logvar is not None:
        with torch.no_grad():
            fixed_logvar = torch.tensor(z_logvar, dtype=torch.float32, device=device)

    global_step = 0
    for cycle in range(n_cycles):

        print(('\n' + '-' * 100) * 2)
        print(f'====> Cycle {cycle + 1}/{n_cycles}')
        writer.add_scalar(tag='progress/cycle', scalar_value=cycle, global_step=global_step)

        for epoch in range(n_epochs):
            global_step += 1

            print('\nTraining...')
            writer.add_scalar(tag='progress/epoch', scalar_value=epoch, global_step=global_step)
            _training_step(model, optimizer, training_data_loader, device, epoch, log_interval=100,
                           writer=writer, global_step=global_step, fixed_logvar=fixed_logvar)

            print('\nEvaluating...')
            _evaluation_step(model, test_data_loader, device, writer=writer, global_step=global_step,
                             fixed_logvar=fixed_logvar)

        _visual_evaluation_step(model, test_set, device, writer=writer, global_step=global_step, cycle_i=cycle,
                                **goal_extractors)

        print('\nSaving checkpoint...')
        model.save(f'{checkpoints_dir}/model_c{cycle}.pt')

        print('Realigning test dataset...')
        test_set.realign_with_model(model)

        print('Realigning training dataset...')
        training_set.realign_with_model(model)

        if new_params_each_cycle:
            print('Resetting params...')
            model.reset_parameters()


def _generate_twin_episodes(n=1_000, max_ep_steps=200, seed=42, render=False):

    import gym
    from gym.envs.robotics import FetchPickAndPlaceEnv
    from gym.envs.robotics import HandPickAndPlaceEnv
    from gym.agents.fetch import FetchPickAndPlaceAgent
    from gym.agents.shadow_hand import HandPickAndPlaceAgent

    def flatten_obs(obs_dict):
        return np.r_[obs_dict['observation'], obs_dict['desired_goal']]

    a_env = gym.make(
        'FetchPickAndPlace-v1'
    ).unwrapped # type: FetchPickAndPlaceEnv

    b_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_id='small_box'
    ).unwrapped # type: HandPickAndPlaceEnv

    a_env.seed(seed)
    b_env.seed(seed)

    a_agent = FetchPickAndPlaceAgent(a_env)
    b_agent = HandPickAndPlaceAgent(b_env)

    a_episodes = []
    b_episodes = []
    prog = tqdm(total=n, desc='Recorded episodes')

    while len(a_episodes) < n:

        a_states = []
        b_states = []

        a_env.reset()
        b_env.reset()

        if a_env.goal[2] < 0.48:
            # desired goal is too close to the table
            continue

        b_env.goal = np.r_[a_env.goal, np.zeros(4)]

        object_pos = b_env.sim.data.get_joint_qpos('object:joint').copy()
        object_pos[:2] = a_env.sim.data.get_joint_qpos('object0:joint')[:2].copy()
        b_env.sim.data.set_joint_qpos('object:joint', object_pos)
        b_env.sim.forward()

        a_agent.reset()
        b_agent.reset()

        a_obs = a_env._get_obs()
        b_obs = b_env._get_obs()

        a_states.append(flatten_obs(a_obs))
        b_states.append(flatten_obs(b_obs))

        success = False

        for _ in range(max_ep_steps):

            a_action = a_agent.predict(a_obs)
            b_action = b_agent.predict(b_obs)

            a_obs, a_rew, a_done, a_info = a_env.step(a_action)
            b_obs, b_rew, b_done, b_info = b_env.step(b_action)

            a_states.append(flatten_obs(a_obs))
            b_states.append(flatten_obs(b_obs))

            if render:
                a_env.render()
                b_env.render()

            success = a_info['is_success'] == 1.0 and b_info['is_success'] == 1.0
            if success:
                break

            if a_done or b_done:
                break

        if success:
            prog.update()
            a_episodes.append(np.array(a_states))
            b_episodes.append(np.array(b_states))

    prog.close()
    return a_episodes, b_episodes


def _generate_twin_episodes_yumi(n=1_000, max_ep_steps=200, seed=42, render=False):

    import gym
    from gym.envs.yumi import YumiConstrainedEnv
    from gym.envs.robotics import HandPickAndPlaceEnv
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    from gym.utils import transformations as tf

    def flatten_obs(obs_dict):
        return np.r_[obs_dict['observation'], obs_dict['desired_goal']]

    a_env = gym.make(
        'YumiConstrained-v1'
    ).unwrapped # type: YumiConstrainedEnv

    b_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_id='box'
    ).unwrapped # type: HandPickAndPlaceEnv

    a_env.seed(seed)
    b_env.seed(seed)

    a_agent = YumiConstrainedAgent(a_env)
    b_agent = HandPickAndPlaceAgent(b_env)

    a_episodes = []
    b_episodes = []
    prog = tqdm(total=n, desc='Recorded episodes')

    a_table_tf = a_env.get_table_surface_pose()
    b_table_tf = b_env.get_table_surface_pose()

    while len(a_episodes) < n:

        a_states = []
        b_states = []

        a_env.reset()
        b_env.reset()

        t_to_goal = tf.get_tf(np.r_[a_env.goal, 1., 0., 0., 0.], a_table_tf)
        b_goal_pose = tf.apply_tf(t_to_goal, b_table_tf)

        b_env.goal = np.r_[b_goal_pose[:3], np.zeros(4)]

        t_to_obj = tf.get_tf(a_env.get_object_pose(), a_table_tf)
        b_obj_pose = tf.apply_tf(t_to_obj, b_table_tf)

        object_pos = b_env.sim.data.get_joint_qpos('object:joint').copy()
        object_pos[:2] = b_obj_pose[:2]
        b_env.sim.data.set_joint_qpos('object:joint', object_pos)
        b_env.sim.forward()

        a_agent.reset()
        b_agent.reset()

        a_obs = a_env._get_obs()
        b_obs = b_env._get_obs()

        a_states.append(flatten_obs(a_obs))
        b_states.append(flatten_obs(b_obs))

        success = False

        for _ in range(max_ep_steps):

            a_action = a_agent.predict(a_obs)
            b_action = b_agent.predict(b_obs)

            a_obs, a_rew, a_done, a_info = a_env.step(a_action)
            b_obs, b_rew, b_done, b_info = b_env.step(b_action)

            a_states.append(flatten_obs(a_obs))
            b_states.append(flatten_obs(b_obs))

            if render:
                a_env.render()
                b_env.render()

            success = a_info['is_success'] == 1.0 and b_info['is_success'] == 1.0
            if success:
                break

            if a_done or b_done:
                break

        if success:
            prog.update()
            a_episodes.append(np.array(a_states))
            b_episodes.append(np.array(b_states))

    prog.close()
    return a_episodes, b_episodes


def _generate_twin_episodes_yumi_reach(n=1_000, max_ep_steps=80, seed=42, render=False):

    import gym
    from gym.envs.yumi import YumiConstrainedEnv
    from gym.envs.robotics import HandPickAndPlaceEnv
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    from gym.utils import transformations as tf

    def flatten_obs(obs_dict):
        return np.r_[obs_dict['observation'], obs_dict['desired_goal']]

    a_env = gym.make(
        'YumiConstrained-v1'
    ).unwrapped # type: YumiConstrainedEnv

    b_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_id='box'
    ).unwrapped # type: HandPickAndPlaceEnv

    a_env.seed(seed)
    b_env.seed(seed)

    a_episodes = []
    b_episodes = []
    prog = tqdm(total=n, desc='Recorded episodes')

    a_table_tf = a_env.get_table_surface_pose()
    b_table_tf = b_env.get_table_surface_pose()

    def hand_action_open(should_open, env):
        u = np.zeros(env.action_space.shape)
        hand_ctrl = u[2:-7]

        hand_ctrl[[0, 3]] = 1.0
        hand_ctrl[[6, 9]] = -1.0

        if should_open:
            hand_ctrl[:] = -1.0
            hand_ctrl[13:] = (-1., -0.5, 1., -1., 0)
        else:
            hand_ctrl[:] = 1.0
            hand_ctrl[13:] = (0.1, 0.5, 1., -1., 0)

        return u

    def yumi_action_open(should_open, env):
        u = np.zeros(env.action_space.shape)
        u[0] = 1. if should_open else -1.
        return u

    def hand_action_reach(obs, env):
        u = np.zeros(env.action_space.shape)
        arm_pos_ctrl = u[-7:-4]
        d = obs['desired_goal'][:3] - env.unwrapped._get_grasp_center_pose()[:3]
        arm_pos_ctrl[:] = d * 2.0
        return u

    def yumi_action_reach(obs, env):
        u = np.zeros(env.action_space.shape)
        arm_pos_ctrl = u[1:4]
        d = obs['desired_goal'][:3] - env.unwrapped.get_grasp_center_pos()
        arm_pos_ctrl[:] = d * 2.0
        return u

    while len(a_episodes) < n:

        a_states = []
        b_states = []

        a_env.reset()
        b_env.reset()

        t_to_goal = tf.get_tf(np.r_[a_env.goal, 1., 0., 0., 0.], a_table_tf)
        b_goal_pose = tf.apply_tf(t_to_goal, b_table_tf)

        b_env.goal = np.r_[b_goal_pose[:3], np.zeros(4)]

        t_to_obj = tf.get_tf(a_env.get_object_pose(), a_table_tf)
        b_obj_pose = tf.apply_tf(t_to_obj, b_table_tf)

        object_pos = b_env.sim.data.get_joint_qpos('object:joint').copy()
        object_pos[:2] = b_obj_pose[:2]
        b_env.sim.data.set_joint_qpos('object:joint', object_pos)
        b_env.sim.forward()

        a_obs = a_env._get_obs()
        b_obs = b_env._get_obs()

        a_states.append(flatten_obs(a_obs))
        b_states.append(flatten_obs(b_obs))

        for i in range(max_ep_steps):

            act = hand_action_open((i // 20) % 2 == 0, b_env)
            b_action = act + hand_action_reach(b_obs, b_env)

            act = yumi_action_open((i // 20) % 2 == 0, a_env)
            a_action = act + yumi_action_reach(a_obs, a_env)

            a_action += np.random.uniform(-1, 1) * 0.01
            b_action += np.random.uniform(-1, 1) * 0.01

            a_obs, a_rew, a_done, a_info = a_env.step(a_action)
            b_obs, b_rew, b_done, b_info = b_env.step(b_action)

            a_states.append(flatten_obs(a_obs))
            b_states.append(flatten_obs(b_obs))

            if render:
                a_env.render()
                b_env.render()

            if a_done or b_done:
                break

        prog.update()
        a_episodes.append(np.array(a_states))
        b_episodes.append(np.array(b_states))

    prog.close()
    return a_episodes, b_episodes


def _generate_twin_episodes_yumi_and_fetch(n=1_000, max_ep_steps=200, seed=42, render=False):

    import gym
    from gym.envs.yumi import YumiConstrainedEnv
    from gym.envs.robotics import FetchPickAndPlaceEnv
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.agents.fetch import FetchPickAndPlaceAgent
    from gym.utils import transformations as tf

    def flatten_obs(obs_dict):
        return np.r_[obs_dict['observation'], obs_dict['desired_goal']]

    a_env = gym.make(
        'YumiConstrained-v1'
    ).unwrapped # type: YumiConstrainedEnv

    b_env = gym.make(
        'FetchPickAndPlace-v1'
    ).unwrapped  # type: FetchPickAndPlaceEnv

    a_env.seed(seed)
    b_env.seed(seed)

    a_agent = YumiConstrainedAgent(a_env)
    b_agent = FetchPickAndPlaceAgent(b_env)

    a_episodes = []
    b_episodes = []
    prog = tqdm(total=n, desc='Recorded episodes')

    a_table_tf = a_env.get_table_surface_pose()
    b_table_tf = gym.make('HandPickAndPlace-v0').unwrapped.get_table_surface_pose()

    while len(a_episodes) < n:

        a_states = []
        b_states = []

        a_env.reset()
        b_env.reset()

        t_to_goal = tf.get_tf(np.r_[a_env.goal, 1., 0., 0., 0.], a_table_tf)
        b_goal_pose = tf.apply_tf(t_to_goal, b_table_tf)

        b_env.goal = b_goal_pose[:3]

        t_to_obj = tf.get_tf(a_env.get_object_pose(), a_table_tf)
        b_obj_pose = tf.apply_tf(t_to_obj, b_table_tf)

        object_pos = b_env.sim.data.get_joint_qpos('object0:joint').copy()
        object_pos[:2] = b_obj_pose[:2]
        b_env.sim.data.set_joint_qpos('object0:joint', object_pos)
        b_env.sim.forward()

        a_agent.reset()
        b_agent.reset()

        a_obs = a_env._get_obs()
        b_obs = b_env._get_obs()

        a_states.append(flatten_obs(a_obs))
        b_states.append(flatten_obs(b_obs))

        success = False

        for _ in range(max_ep_steps):

            a_action = a_agent.predict(a_obs)
            b_action = b_agent.predict(b_obs)

            a_obs, a_rew, a_done, a_info = a_env.step(a_action)
            b_obs, b_rew, b_done, b_info = b_env.step(b_action)

            a_states.append(flatten_obs(a_obs))
            b_states.append(flatten_obs(b_obs))

            if render:
                a_env.render()
                b_env.render()

            success = a_info['is_success'] == 1.0 and b_info['is_success'] == 1.0
            if success:
                break

            if a_done or b_done:
                break

        if success:
            prog.update()
            a_episodes.append(np.array(a_states))
            b_episodes.append(np.array(b_states))

    prog.close()
    return a_episodes, b_episodes


def _generate_twin_dataset(file_path, n=1_000, max_ep_steps=200, seed=42, render=False, generator=None):
    generator = generator or _generate_twin_episodes
    a_episodes, b_episodes = generator(n=n, max_ep_steps=max_ep_steps, seed=seed, render=render)
    dataset = TwinDataset(a_episodes=a_episodes, b_episodes=b_episodes, copy=False)
    dataset.save(file_path, with_aligned_states=False)


def _test_dwt_on_dataset(dataset: TwinDataset):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.metrics.pairwise import euclidean_distances

    def distance_fn(a, b):
        a_obj_pos = a[3:6] # fetch env
        b_obj_pos = b[63:66] # hand env
        return np.linalg.norm(a_obj_pos - b_obj_pos)

    def plot_obj_traj(a, b):
        from mpl_toolkits.mplot3d import Axes3D
        a_obj_traj = a[:, 3:6] # fetch env
        b_obj_traj = b[:, 63:66] # hand env
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*a_obj_traj.T, label='Fetch')
        ax.plot(*b_obj_traj.T, label='Hand')
        ax.legend()
        plt.show()

    for a_ep, b_ep in zip(dataset.a_episodes, dataset.b_episodes):
        plot_obj_traj(a_ep, b_ep)
        cost, path = dynamic_time_warping_py(a_ep, b_ep, distance_fn)
        plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
        plt.plot(*path.T, 'w')
        plt.show()


def _test_reconstruction(dataset: TwinDataset, model: TwinVAE):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    def plot_obj_traj(a, b):
        from mpl_toolkits.mplot3d import Axes3D
        # a_obj_traj = a[:, 63:66] # hand env
        # b_obj_traj = b[:, 63:66]
        a_obj_traj = a[:, 3:6] # fetch env
        b_obj_traj = b[:, 3:6]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*a_obj_traj.T, label='orig')
        ax.plot(*b_obj_traj.T, label='recon')
        ax.legend()
        plt.show()

    dev = torch.device('cuda')
    model.to(dev)

    for a_ep, b_ep in zip(dataset.a_episodes, dataset.b_episodes):
        with torch.no_grad():
            ep = torch.tensor(a_ep, dtype=torch.float32, device=dev)
            recon = model.a_vae.forward(ep)[1].cpu().numpy()
        plot_obj_traj(a_ep, recon)


def _test_sim_loss(dataset: TwinDataset, model: TwinVAE, a_get_ag, a_get_g, b_get_ag, b_get_g):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    dev = torch.device('cuda')
    model.to(dev)

    for a_ep, b_ep in zip(dataset.a_episodes, dataset.b_episodes):
        with torch.no_grad():
            a_ep_t = torch.tensor(a_ep, dtype=torch.float32, device=dev)
            b_ep_t = torch.tensor(b_ep, dtype=torch.float32, device=dev)
            n, m = a_ep_t.shape[0], b_ep_t.shape[0]
            res = torch.zeros((n, m), dtype=torch.float32, device=dev)
            for i in range(n):
                sim_loss = model.compute_elementwise_sim_loss(a_ep_t[i].repeat(m, 1), b_ep_t, reparameterize=False)
                res[i] = sim_loss.mean(dim=1)
            res = res.cpu().numpy()

        sim = np.diag(res)
        cost, path = dynamic_time_warping_c(res)
        a_ep_s = dataset.a_scaler.inverse_transform(a_ep, copy=True)
        b_ep_s = dataset.b_scaler.inverse_transform(b_ep, copy=True)
        a_dtw_s = a_ep_s[path[:, 0]]
        b_dtw_s = b_ep_s[path[:, 1]]
        sim_dtw = res[path[:, 0], path[:, 1]]

        a_ag = a_get_ag(a_ep_s)
        a_g = a_get_g(a_ep_s)
        b_ag = b_get_ag(b_ep_s)
        b_g = b_get_g(b_ep_s)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        axes[0, 0].imshow(res, origin='lower', cmap=cm.gist_stern, interpolation='nearest')
        axes[0, 0].plot(path[:, 1], path[:, 0], 'w')

        a = np.linalg.norm(a_ag - a_g, axis=1)
        b = np.linalg.norm(b_ag - b_g, axis=1)
        axes[1, 0].plot(a, label="A")
        axes[1, 0].plot(b, label="B")
        axes[1, 0].plot(-sim, label="sim")
        axes[1, 0].plot(-np.abs(a - b), label="diff")
        axes[1, 0].grid()
        axes[1, 0].legend()

        axes[0, 1].imshow(cost, origin='lower', cmap=cm.gist_stern, interpolation='nearest')
        axes[0, 1].plot(path[:, 1], path[:, 0], 'w')

        a = np.linalg.norm(a_get_ag(a_dtw_s) - a_get_g(a_dtw_s), axis=1)
        b = np.linalg.norm(b_get_ag(b_dtw_s) - b_get_g(b_dtw_s), axis=1)
        axes[1, 1].plot(a, label="A")
        axes[1, 1].plot(b, label="B")
        axes[1, 1].plot(-sim_dtw, label="sim")
        axes[1, 1].plot(-np.abs(a - b), label="diff")
        axes[1, 1].grid()
        axes[1, 1].legend()

        plt.show()


if __name__ == '__main__':
    # _generate_twin_dataset(file_path='../out/pp_twin_dataset_10k.pkl', n=10_000, render=False)
    # _generate_twin_dataset(file_path='../out/pp_yumi_twin_dataset_3k.pkl', n=3_000, render=False,
    #                        generator=_generate_twin_episodes_yumi, max_ep_steps=110)
    # _generate_twin_dataset(file_path='../out/pp_reach_yumi_twin_dataset_2k.pkl', n=2_000, render=False,
    #                        generator=_generate_twin_episodes_yumi_reach, max_ep_steps=80)
    # _generate_twin_dataset(file_path='../out/pp_yumi_fetch_twin_dataset_5k.pkl', n=5_000, render=False,
    #                        generator=_generate_twin_episodes_yumi_and_fetch, max_ep_steps=80)
    # exit(0)

    # _full_dataset = TwinDataset.load('../out/pp_twin_dataset_10k.pkl')
    # _full_dataset = TwinDataset.load('../out/pp_yumi_fetch_twin_dataset_5k.pkl')
    _full_dataset = TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl')
    _secondary_ds = TwinDataset.load('../out/pp_reach_yumi_twin_dataset_2k.pkl')
    _full_dataset = TwinDataset.merge(_full_dataset, _secondary_ds)
    _full_dataset.normalize()

    # _test_dwt_on_dataset(_full_dataset)
    # exit(0)

    # _test_reconstruction(_full_dataset, model=TwinVAE.load('../out/pp_twin_simple_test/checkpoints/model_c2.pt',
    #                                                        net_class=SimpleAutoencoder))
    # exit(0)

    # _test_sim_loss(
    #     _full_dataset,
    #     model=TwinVAE.load('../out/pp_twin_ae_test/checkpoints/model_c9.pt',
    #                        net_class=SimpleAutoencoder),
    #     a_get_ag=lambda x: x[:, 3:6],
    #     b_get_ag=lambda x: x[:, 63:66],
    #     a_get_g=lambda x: x[0, -7:-4],
    #     b_get_g=lambda x: x[0, -7:-4],
    # )

    # _test_sim_loss(
    #     _full_dataset,
    #     model=TwinVAE.load('../out/pp_and_reach_yumi_twin_ae_test_z15/checkpoints/model_c49.pt',
    #                        net_class=SimpleAutoencoder),
    #     a_get_ag=lambda x: x[:, 18:21],
    #     b_get_ag=lambda x: x[:, 63:66],
    #     a_get_g=lambda x: x[0, -3:],
    #     b_get_g=lambda x: x[0, -7:-4],
    # )
    # exit(0)

    _training_set, _test_set = _full_dataset.split(
        shuffle=True,
        test_size=0.25,
        train_size=None,
        seed=42,
    )

    train(
        training_set=_training_set,
        test_set=_test_set,
        local_dir='../out/twin_yumi_hand_ae_resets_REMOVE',
        net_class=SimpleAutoencoder,
        new_params_each_cycle=True,
        n_epochs=20,
        loss_weights=dict(
            a=(1. / _training_set.a_item_size),
            b=(1. / _training_set.b_item_size),
            sim=2.0,
        ),
        goal_extractors=dict(
            a_get_ag=lambda x: x[:, 18:21],
            b_get_ag=lambda x: x[:, 63:66],
            a_get_g=lambda x: x[0, -3:],
            b_get_g=lambda x: x[0, -7:-4],
        ),
        init_with_goal_based_alignment=True,
    )

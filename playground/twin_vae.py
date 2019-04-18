import pickle
from typing import Callable
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dtw import dynamic_time_warping


class VAE(nn.Module):

    # From : https://github.com/pytorch/examples/blob/5df464c46cf321ed1cc3df1e670358d7f5ae1887/vae/main.py#L39
    def __init__(self, input_size: int, hidden_size: int, z_size: int):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, z_size)
        self.fc22 = nn.Linear(hidden_size, z_size)
        self.fc3 = nn.Linear(z_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.input_size = input_size

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def build_loss_function(self):
        input_size = self.input_size

        # From VAE example code
        # Reconstruction + KL divergence losses summed over all elements and batch
        def loss_function(x, recon_x, mu, logvar):
            bce = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction='sum')
            kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
            return bce + kdl

        return loss_function


class TwinVAE:

    def __init__(self, a_spec: dict, b_spec: dict, z_size: int):
        self.a_vae = VAE(**a_spec, z_size=z_size)
        self.b_vae = VAE(**b_spec, z_size=z_size)

        self.a_loss = self.a_vae.build_loss_function()
        self.b_loss = self.b_vae.build_loss_function()
        self.sim_loss = nn.MSELoss()
        self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    def compute_losses(self, a_data, b_data):

        a_z, a_recon_x, a_mu, a_logvar = self.a_vae(a_data)
        a_loss = self.a_loss(a_data, a_recon_x, a_mu, a_logvar)

        b_z, b_recon_x, b_mu, b_logvar = self.b_vae(b_data)
        b_loss = self.b_loss(b_data, b_recon_x, b_mu, b_logvar)

        sim_loss = self.sim_loss(a_z, b_z)
        total_loss = a_loss + b_loss + sim_loss

        return total_loss, a_loss, b_loss, sim_loss

    def to(self, device):
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


def _training_step(model: TwinVAE, optimizer: optim.Optimizer, data_loader, device, epoch: int, log_interval=1):
    model.train()
    train_loss = 0

    for batch_idx, (a_data, b_data) in enumerate(data_loader):

        a_data = a_data.to(device)
        b_data = b_data.to(device)

        optimizer.zero_grad()
        losses = model.compute_losses(a_data, b_data)
        total_loss, a_loss, b_loss, sim_loss = losses

        total_loss.backward()

        train_loss += total_loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(a_data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), total_loss.item() / len(a_data))
            )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))


def _evaluation_step(model: TwinVAE, data_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (a_data, b_data) in enumerate(data_loader):

            a_data = a_data.to(device)
            b_data = b_data.to(device)

            total_loss = model.compute_losses(a_data, b_data)[0]
            test_loss += total_loss.item()

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


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

    def __getitem__(self, idx):
        return self.a_states[idx].astype(np.float32), self.b_states[idx].astype(np.float32)

    def __len__(self):
        return len(self.a_states)

    def split(self, shuffle=True, test_size=0.25, seed=None, copy=False):
        a_train, a_test, b_train, b_test = train_test_split(self.a_episodes, self.b_episodes, test_size=test_size,
                                                            shuffle=shuffle, random_state=seed)
        return (
            TwinDataset(a_train, b_train, copy=copy),
            TwinDataset(a_test, b_test, copy=copy),
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

    def assume_temporal_alignment(self):
        self.realign(lambda x, y: 0.0)

    def realign(self, distance_fn: Callable):
        self.a_states = []
        self.b_states = []
        for a_ep, b_ep in tqdm(zip(self.a_episodes, self.b_episodes),
                               desc='Aligned episodes', total=len(self.a_episodes)):
            cost, path = dynamic_time_warping(a_ep, b_ep, distance_fn)
            self.a_states += list(a_ep[path[:, 0]])
            self.b_states += list(b_ep[path[:, 1]])

    def realign_with_model(self, model: TwinVAE):
        model.eval()
        dev = model.device

        def distance_fn(a, b):
            a = torch.tensor(a[np.newaxis], requires_grad=False, dtype=torch.float32, device=dev)
            b = torch.tensor(b[np.newaxis], requires_grad=False, dtype=torch.float32, device=dev)
            sim_loss = model.compute_losses(a, b)[-1]
            return sim_loss.item()

        with torch.no_grad():
            self.realign(distance_fn)

    @property
    def a_item_size(self):
        return self.a_episodes[0].shape[1]

    @property
    def b_item_size(self):
        return self.b_episodes[0].shape[1]


def train(*, training_set: TwinDataset, test_set: TwinDataset, local_dir: str, seed=42,
          n_workers=1, n_epochs=100, batch_size=32):

    torch.manual_seed(seed)

    model = TwinVAE(
        a_spec=dict(input_size=training_set.a_item_size, hidden_size=32),
        b_spec=dict(input_size=training_set.b_item_size, hidden_size=16),
        z_size=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Initial alignment of datasets...')
    training_set.assume_temporal_alignment()
    test_set.assume_temporal_alignment()

    training_data_loader = DataLoader(dataset=training_set, num_workers=n_workers, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):

        print('Training...')
        _training_step(model, optimizer, training_data_loader, device, epoch)

        print('Realigning test dataset...')
        test_set.realign_with_model(model)

        print('Evaluating...')
        _evaluation_step(model, test_data_loader, device)

        print('Realigning training dataset...')
        training_set.realign_with_model(model)


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


def _generate_twin_dataset(file_path, n=1_000, max_ep_steps=200, seed=42, render=False):
    a_episodes, b_episodes = _generate_twin_episodes(n=n, max_ep_steps=max_ep_steps, seed=seed, render=render)
    dataset = TwinDataset(a_episodes=a_episodes, b_episodes=b_episodes, copy=False)
    dataset.save(file_path, with_aligned_states=False)


if __name__ == '__main__':
    # _generate_twin_dataset(file_path='../out/pp_twin_dataset.pkl', n=100, render=False)

    _full_dataset = TwinDataset.load('../out/pp_twin_dataset.pkl')
    _training_set, _test_set = _full_dataset.split(shuffle=True, test_size=0.25, seed=42)

    train(
        training_set=_training_set,
        test_set=_test_set,
        local_dir='../out/pp_twin_test'
    )

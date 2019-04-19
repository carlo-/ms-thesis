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
                sim_loss = model.compute_elementwise_sim_loss(a[i].repeat(m, 1), b)
                res[i] = sim_loss.mean(dim=1)
            precomputed_distances = res.cpu()

        cost, path = dynamic_time_warping(a_ep, b_ep, precomputed_distances=precomputed_distances)
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

    def reset_parameters(self):
        def reset(m):
            if m != self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset)

    def _encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self._encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self._encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    # From VAE example code
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, recon_x, mu, logvar):
        bce = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), reduction='sum')
        kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
        return bce + kdl


class TwinVAE:

    def __init__(self, a_spec: dict, b_spec: dict, z_size: int):
        self.a_spec = dict(a_spec)
        self.b_spec = dict(b_spec)
        self.z_size = z_size

        self.a_vae = VAE(**a_spec, z_size=z_size)
        self.b_vae = VAE(**b_spec, z_size=z_size)

        self.a_loss = self.a_vae.loss_function
        self.b_loss = self.b_vae.loss_function
        self.sim_loss = nn.MSELoss()
        self.sim_loss_no_red = nn.MSELoss(reduction='none')
        self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    def reset_parameters(self):
        self.a_vae.reset_parameters()
        self.b_vae.reset_parameters()

    def compute_elementwise_sim_loss(self, a_data, b_data):
        a_z = self.a_vae.encode(a_data)
        b_z = self.b_vae.encode(b_data)
        return self.sim_loss_no_red(a_z, b_z)

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

    def clone(self):
        c = TwinVAE(self.a_spec, self.b_spec, self.z_size)
        c.a_vae.load_state_dict(self.a_vae.state_dict())
        c.b_vae.load_state_dict(self.b_vae.state_dict())
        c.to(self.device)
        return c


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

    def split(self, shuffle=True, test_size=0.25, train_size=None, seed=None, copy=False):
        a_train, a_test, b_train, b_test = train_test_split(self.a_episodes, self.b_episodes, test_size=test_size,
                                                            train_size=train_size, shuffle=shuffle, random_state=seed)
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

    def _extract_states(self):
        # equivalent to self.realign(lambda x, y: 0.0)
        self.a_states = []
        self.b_states = []
        for a_ep, b_ep in tqdm(zip(self.a_episodes, self.b_episodes),
                               desc='Processed episodes', total=len(self.a_episodes)):
            min_len = min(len(a_ep), len(b_ep))
            self.a_states += list(a_ep[:min_len])
            self.b_states += list(b_ep[:min_len])

    def preprocess(self):
        if len(self.a_states) == 0 or len(self.b_states) != len(self.a_states):
            self._extract_states()

    def realign(self, distance_fn: Callable):
        self.a_states = []
        self.b_states = []
        for a_ep, b_ep in tqdm(zip(self.a_episodes, self.b_episodes),
                               desc='Aligned episodes', total=len(self.a_episodes)):
            cost, path = dynamic_time_warping(a_ep, b_ep, distance_fn)
            self.a_states += list(a_ep[path[:, 0]])
            self.b_states += list(b_ep[path[:, 1]])

    def realign_with_model(self, model: TwinVAE, workers=7):

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
            print(f'Worker {worker_i} done.')

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


def train(*, training_set: TwinDataset, test_set: TwinDataset, local_dir: str, seed=42,
          n_workers=1, n_epochs=100, batch_size=256):

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
    training_set.preprocess()
    test_set.preprocess()

    training_data_loader = DataLoader(dataset=training_set, num_workers=n_workers, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):

        print('Training...')
        _training_step(model, optimizer, training_data_loader, device, epoch, log_interval=100)

        print('Evaluating...')
        _evaluation_step(model, test_data_loader, device)

        print('Realigning test dataset...')
        test_set.realign_with_model(model)

        print('Realigning training dataset...')
        training_set.realign_with_model(model)

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


def _generate_twin_dataset(file_path, n=1_000, max_ep_steps=200, seed=42, render=False):
    a_episodes, b_episodes = _generate_twin_episodes(n=n, max_ep_steps=max_ep_steps, seed=seed, render=render)
    dataset = TwinDataset(a_episodes=a_episodes, b_episodes=b_episodes, copy=False)
    dataset.save(file_path, with_aligned_states=False)


if __name__ == '__main__':
    # _generate_twin_dataset(file_path='../out/pp_twin_dataset_10k.pkl', n=10_000, render=False)

    _full_dataset = TwinDataset.load('../out/pp_twin_dataset_10k.pkl')
    _training_set, _test_set = _full_dataset.split(
        shuffle=True,
        test_size=0.25,
        train_size=None,
        seed=42
    )

    train(
        training_set=_training_set,
        test_set=_test_set,
        local_dir='../out/pp_twin_test'
    )

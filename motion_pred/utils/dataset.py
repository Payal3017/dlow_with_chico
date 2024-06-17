import numpy as np


class Dataset:

    def __init__(self, mode, t_his, t_pred, actions="all", kept_joints=np.arange(15), collision=False, data_path="chico\\data\\CHICO\\dataset", win_stride=1,dataset="chico"):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.collision = collision
        self.data_path = data_path
        self.win_stride = win_stride
        self.win_size = t_his + t_pred
        self.prepare_data()
        self.std, self.mean = None, None
        self.dataset = dataset

        if dataset == "chico":
            self.data_len = self.data.shape[0]

        else:
            self.data_len = sum(
                [seq.shape[0] for data_s in self.data.values() for seq in data_s.values()]
            )
        self.kept_joints = kept_joints
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def sample(self):
        if self.dataset == "chico":
            idx = np.random.randint(self.data.shape[0])
            traj = self.data[idx]
            return traj[None, ...]
        else:
            subject = np.random.choice(self.subjects)
            dict_s = self.data[subject]
            action = np.random.choice(list(dict_s.keys()))
            seq = dict_s[action]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start:fr_end]
            return traj[None, ...]

    def sampling_generator(self, num_samples=1000, batch_size=8):
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            yield sample

    def iter_generator(self, step=25):
        if self.dataset == "chico":
            seq_len = self.data.shape[0]
            for i in range(0, seq_len - self.t_total, step):
                traj = self.data[None, i : i + self.t_total]
                yield traj
        else:
            for data_s in self.data.values():
                for seq in data_s.values():
                    seq_len = seq.shape[0]
                    for i in range(0, seq_len - self.t_total, step):
                        traj = seq[None, i : i + self.t_total]
                        yield traj

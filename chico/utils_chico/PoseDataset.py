import os
import numpy as np
import pickle as pk
from chico.utils_chico.data_utils import (
    abnorm_actions_,
    normal_actions,
    normal_actions_,
    abnorm_actions,
)
from chico.utils_chico.data_utils import downs_from_22, get_dct_matrix, dct_transform_numpy
from torch.utils.data import Dataset
from motion_pred.utils.dataset import Dataset as Dataset2
class PoseDatasetDlow(Dataset2):

    def __init__(self, mode, data_path="CHICO", t_his=10, t_pred=25, actions='all', win_stride=0, collision=False,dataset="chico", **kwargs):
        print(f" in_tf {t_his} out_tf {t_pred}")
        super().__init__(mode, t_his, t_pred, actions, dataset=dataset)
        self.collision = collision
        self.data_path = os.path.join("chico", "data", "CHICO", "dataset")
        self.t_his = t_his
        self.t_pred = t_pred
        self.in_tf = t_his
        self.out_tf = t_pred
        self.win_size = t_his + t_pred
        self.win_stride = self.win_size if win_stride == 0 else win_stride
        self.prepare_data()

    def prepare_data(self):
        self.subjects_split = {
            'train': ["S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S01", "S16", "S17"],
            'validation': ["S00", "S04"],
            'test': ["S02", "S03", "S18", "S19"]
        }
        self.subjects = self.subjects_split[self.mode]

        if self.collision:
            self.allowed_actions = {"file": abnorm_actions, "acts": abnorm_actions_}
        else:
            self.allowed_actions = {"file": normal_actions, "acts": normal_actions_}

        self.actions = [act for act in self.allowed_actions['file']] if self.actions == 'all' else self.actions
        self.process_data()

    def process_data(self):
        all_data = []

        for subject in self.subjects:
            sub_path = os.path.join(self.data_path, subject)
            sub_actions_paths = [os.path.join(sub_path, act) for act in os.listdir(sub_path) if act in self.actions]

            for sub_actions_path in sub_actions_paths:
                data = self.retrieve_data(sub_actions_path)
                splitted_windows = self.split_single_pose(data)
                all_data.append(splitted_windows)

        self.data = np.concatenate(all_data)

    def retrieve_data(self, action_path):
        with open(action_path, "rb") as f:
            all_data = pk.load(f)

        human_related_data = [x[0] for x in all_data]
        single_hpose_np = np.stack(human_related_data, axis=0)

        return single_hpose_np

    def split_single_pose(self, single_pose_array):
        T, _, _ = single_pose_array.shape
        iterations = np.ceil((T - self.win_size) / self.win_stride).astype(int) + 1
        all_windows = []
        for segment in range(iterations):
            start_index = self.win_stride * segment
            end_index = start_index + self.win_size
            curr_win = single_pose_array[start_index:end_index]
            if curr_win.shape[0] == (self.t_his + self.t_pred):
                all_windows.append(curr_win)

        return np.stack(all_windows, axis=0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class PoseDataset(Dataset):
    def __init__(
        self,
        data_path="CHICO",
        split="train",
        input_time_frames=10,
        output_time_frames=25,
        win_stride=0,
        collision=False,
        actions=None,
    ):
        super(PoseDataset).__init__()

        self.data_path = data_path
        self.split = split
        self.in_tf = input_time_frames
        self.out_tf = output_time_frames
        self.win_size = input_time_frames + output_time_frames
        self.win_stride = self.win_size if win_stride == 0 else win_stride

        if split == "train":
            # self.subjects = ['S00','S01','S02', 'S03', 'S04', 'S05', 'S06', 'S07','S08',
            #                    'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']

            self.subjects = [
                "S05",
                "S06",
                "S07",
                "S08",
                "S09",
                "S10",
                "S11",
                "S12",
                "S13",
                "S14",
                "S15",
                "S01",
                "S16",
                "S17",
            ]

            if collision == False:
                self.allowed_actions = {"file": normal_actions, "acts": normal_actions_}
            else:
                self.allowed_actions = {"file": abnorm_actions, "acts": abnorm_actions_}
        elif split == "validation":

            self.subjects = ["S00", "S04"]  # ,'S01', 'S16', 'S17']#[ 'S16', 'S17']
            if collision == False:
                self.allowed_actions = {"file": normal_actions, "acts": normal_actions_}
            else:
                self.allowed_actions = {"file": abnorm_actions, "acts": abnorm_actions_}
        elif split == "test":
            # self.subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']

            self.subjects = ["S02", "S03", "S18", "S19"]  # ['S18','S19']

            if collision == False:
                self.allowed_actions = {
                    "file": normal_actions,  # abnorm_actions,
                    "acts": normal_actions_,
                }  # abnorm_actions_}
            else:
                self.allowed_actions = {"file": abnorm_actions, "acts": abnorm_actions_}
        else:
            self.subjects = None
        """
        if split == 'train':
            self.subjects = ['S04', 'S05', 'S06', 'S07', 'S08',
                                'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
            self.allowed_actions = {'file': normal_actions,
                                    'acts': normal_actions_}
        elif split == 'validation':
            self.subjects = ['S00']#,'S01', 'S16', 'S17']
            self.allowed_actions = {'file': normal_actions,
                                    'acts': normal_actions_}
        elif split == 'test':
            #self.subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']

            self.subjects = ['S02', 'S03', 'S18', 'S19']
            self.allowed_actions = {'file': normal_actions,
                                    'acts': normal_actions_}"""

        # self.allowed_actions = {'file': normal_actions,
        #                         'acts': normal_actions_}
        self.actions = []

        for act in actions:
            if act.endswith(".pkl"):
                if act in self.allowed_actions["file"]:
                    self.actions.append(act)
            else:
                if act in self.allowed_actions["acts"]:
                    self.actions.append(act + ".pkl")

        assert self.actions != [], "there should be at least one action"

        self.windows = self.build_dataset()

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self):
        return self.windows.shape[0]

    def build_dataset(self):

        all_data = []

        for subject in self.subjects:
            sub_path = os.path.join(self.data_path, subject)
            sub_actions_paths = [
                os.path.join(sub_path, act)
                for act in os.listdir(sub_path)
                if ((act.endswith(".pkl")) & (act in self.actions))
            ]
            for sub_actions_path in sub_actions_paths:
                data = self.retrieve_data(sub_actions_path)
                splitted_windows = self.split_single_pose(data)
                all_data.append(splitted_windows)
        return np.concatenate(all_data)

    def retrieve_data(self, action_path):
        with open(action_path, "rb") as f:
            all_data = pk.load(f)

        human_related_data = [x[0] for x in all_data]

        single_hpose_np = np.stack(human_related_data, axis=0)

        return single_hpose_np

    def split_single_pose(self, single_pose_array):
        T, _, _ = single_pose_array.shape

        iterations = np.ceil((T - self.win_size) / self.win_stride).astype(int) + 1
        all_windows = []
        for segment in range(iterations):
            start_index = self.win_stride * segment
            end_index = start_index + self.win_size
            print(f"start_index: {start_index}, end_index: {self.win_size}")
            print(f"win_strade: {self.win_stride} \n segment: {segment}")
            print(start_index, end_index)
            curr_win = single_pose_array[start_index:end_index]
            if curr_win.shape[0] == (self.in_tf + self.out_tf):
                all_windows.append(curr_win)

        return np.stack(all_windows, axis=0)


class PoseDatasetMSR(Dataset):
    def __init__(
        self,
        data_path="safeHR",
        split="train",
        input_time_frames=10,
        output_time_frames=25,
        win_stride=0,
        actions=None,
        dct_used=15,
        global_min=0,
        global_max=0,
        debug_step=100,
    ):
        super(PoseDataset).__init__()

        self.data_path = data_path
        self.split = split
        self.in_tf = input_time_frames
        self.out_tf = output_time_frames
        self.win_size = input_time_frames + output_time_frames
        self.win_stride = self.win_size if win_stride == 0 else win_stride

        if split == "train":
            self.subjects = ["S02", "S03", "S04", "S05", "S06", "S08"]
        elif split == "validation":
            self.subjects = ["S07"]
        elif split == "test":
            self.subjects = ["S01"]
        else:
            self.subjects = None

        self.allowed_actions = {"file": normal_actions, "acts": normal_actions_}
        self.actions = []

        for act in actions:
            if act.endswith(".pkl"):
                if act in self.allowed_actions["file"]:
                    self.actions.append(act)
            else:
                if act in self.allowed_actions["acts"]:
                    self.actions.append(act + ".pkl")

        assert self.actions != [], "there should be at least one action"

        self.windows = self.build_dataset()

        ###### MSR #####

        self.Index1714 = [
            [0, 9],
            [1],
            [2],
            [3],  ##r_leg
            [4],
            [5],
            [6],  # l_leg
            [7, 8, 10],  # head
            [11],
            [12],
            [13],  # l_arm
            [14],
            [15],
            [16],
        ]  # r_arm

        self.Index149 = [
            [0, 7],  # head-body
            [1, 2],
            [2, 3],  # r_leg
            [4, 5],
            [5, 6],  # l_leg
            [8, 9],
            [9, 10],  # l_arm
            [11, 12],
            [12, 13],
        ]  # r_arm

        self.Index94 = [
            [0, 1, 2],  # down_right
            [0, 3, 4],  # down_left
            [0, 5, 6],  # upper_left
            [0, 7, 8],
        ]  # upper_right

        self.down_key = [
            ("p17", "p14", self.Index1714),
            ("p14", "p9", self.Index149),
            ("p9", "p4", self.Index94),
        ]

        S, T, V, D = self.windows.shape
        gt_17 = self.windows.transpose(0, 2, 3, 1)
        gt_17 = gt_17.reshape(S, D * V, T)

        gt_all_scales = {"p17": gt_17}
        gt_all_scales = downs_from_22(gt_all_scales, down_key=self.down_key)

        input_all_scales = {}
        for k in gt_all_scales.keys():
            input_all_scales[k] = np.concatenate(
                (
                    gt_all_scales[k][:, :, : self.in_tf],
                    np.repeat(
                        gt_all_scales[k][:, :, self.in_tf - 1 : self.in_tf],
                        self.out_tf,
                        axis=-1,
                    ),
                ),
                axis=-1,
            )

        self.dct_used = dct_used
        self.dct_m, self.idct_m = get_dct_matrix(self.win_size)

        for k in input_all_scales:
            input_all_scales[k] = dct_transform_numpy(
                input_all_scales[k], self.dct_m, dct_used
            )

        # Max min norm to -1 -> 1 ***********
        self.global_max = global_max
        self.global_min = global_min

        if split == "train":
            gt_max = []
            gt_min = []
            for k in gt_all_scales.keys():
                gt_max.append(np.max(gt_all_scales[k]))
                gt_min.append(np.min(gt_all_scales[k]))
            for k in input_all_scales.keys():
                gt_max.append(np.max(input_all_scales[k]))
                gt_min.append(np.min(input_all_scales[k]))

            self.global_max = np.max(np.array(gt_max))
            self.global_min = np.min(np.array(gt_min))

        for k in input_all_scales.keys():
            input_all_scales[k] = (input_all_scales[k] - self.global_min) / (
                self.global_max - self.global_min
            )
            input_all_scales[k] = input_all_scales[k] * 2 - 1

        # todo 加速调试 *********************************
        little = np.arange(
            0, input_all_scales[list(input_all_scales.keys())[0]].shape[0], debug_step
        )
        for k in input_all_scales:
            input_all_scales[k] = input_all_scales[k][little]
            gt_all_scales[k] = gt_all_scales[k][little]

        self.gt_all_scales = gt_all_scales
        self.input_all_scales = input_all_scales
        ########################

    def __getitem__(self, index):
        gts = {}
        inputs = {}
        for k in ["p17", "p14", "p9", "p4"]:
            gts[k] = self.gt_all_scales[k][index]
            inputs[k] = self.input_all_scales[k][index]

        return inputs, gts

    def __len__(self):
        return self.gt_all_scales[list(self.gt_all_scales.keys())[0]].shape[0]

    def build_dataset(self):

        all_data = []

        for subject in self.subjects:
            sub_path = os.path.join(self.data_path, subject)
            sub_actions_paths = [
                os.path.join(sub_path, act)
                for act in os.listdir(sub_path)
                if ((act.endswith(".pkl")) & (act in self.actions))
            ]
            for sub_actions_path in sub_actions_paths:
                data = self.retrieve_data(sub_actions_path)
                splitted_windows = self.split_single_pose(data)
                all_data.append(splitted_windows)

        np.savez(f"npppp_{self.split}_dataset.npz", positions_3d=all_data)
        return np.concatenate(all_data)

    def retrieve_data(self, action_path):
        with open(action_path, "rb") as f:
            all_data = pk.load(f)

        human_related_data = [x[0] for x in all_data]

        single_hpose_np = np.stack(human_related_data, axis=0)

        return single_hpose_np

    def split_single_pose(self, single_pose_array):
        T, _, _ = single_pose_array.shape

        iterations = np.ceil((T - self.win_size) / self.win_stride).astype(int) + 1
        all_windows = []
        for segment in range(iterations):
            start_index = self.win_stride * segment
            end_index = start_index + self.win_size

            curr_win = single_pose_array[start_index:end_index]
            all_windows.append(curr_win)

        return np.stack(all_windows, axis=0)

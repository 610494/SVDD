from torch.utils.data import Dataset, Subset
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx
import numpy as np
import os
import torch
import pandas as pd


class LangEmbDataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class: str):  # Original: __init__(self, normal_class: str)
        super().__init__(root)
        # print(f"************\n[datasets/lang_emb.py] root:{self.root}\n************") # root: None
        # print(f"************\n[datasets/lang_emb.py] normal_classes:{self.normal_classes}\n************") # None

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])

        train_set = MyLangEmb(npz_file_path=os.path.join( # to change for train!
            '/Users/ca/Desktop/SVDD/df_train_clean.npz'), normal_class=normal_class)
        self.train_set = train_set

        test_set = MyLangEmb(npz_file_path=os.path.join( # to change for test and inference!
            '/Users/ca/Desktop/SVDD/df_test_clean.npz'), normal_class=normal_class)
        self.test_set = test_set


class MyLangEmb(Dataset):
    def __init__(self, npz_file_path: str, normal_class: str):
        super().__init__()
        # if self.features: print(f"************\n[datasets/lang_emb.py] features:{self.features}\n************") # features: [[0. 0. 0. ... 0. 0. 0.]]
        # if self.ids: print(f"************\n[datasets/lang_emb.py] ids:{self.ids}\n************") # ids: [0]
        # if self.npz_file: print(f"************\n[datasets/lang_emb.py] npz_file:{self.npz_file}\n************") # npz_file: <numpy.lib.npyio.NpzFile object at 0x7f9b1c0b6a90>
        # if self.normal_classes: print(f"************\n[datasets/lang_emb.py] normal_classes:{self.normal_classes}\n************") # normal_classes: ['en']
        
        self.npz_file = np.load(npz_file_path, allow_pickle=True)
        self.normal_classes = []
        self.ids = self.npz_file["ids"].copy()
        self.features = self.npz_file["features"].copy()

        # self.lang_emb_list = self.npz_file["lang_emb"].copy()
        # self.lang_kind_list = self.npz_file["lang_kind"].copy()
        # self.filename = self.npz_file["filename"].copy()
        # self.outlier_classes = list(np.unique(self.lang_kind_list))

        # for class_name in self.outlier_classes:
        #     if class_name.split("_")[0] == normal_class:
        #         self.outlier_classes.remove(class_name)
        #         self.normal_classes.append(class_name)

    # def __getitem__(self, index):
    #     # lang_emb = torch.from_numpy(self.lang_emb_list[index])
    #     lang_emb = lang_emb.unsqueeze(0)
    #     # return lang_emb, int(self.lang_kind_list[index] in self.outlier_classes), index, self.lang_kind_list[index], self.filename[index]

    def __getitem__(self, index):
        id_value = self.ids[index]
        feature_values = torch.from_numpy(self.features[index])
        feature_values = feature_values.unsqueeze(0)
        return feature_values, id_value

    def __len__(self):
        # return len(self.npz_file["lang_emb"])
        return len(self.npz_file["features"])

    # def train_labels(self) -> np.ndarray:
    #     return np.asarray([lang_kind.split("_")[0] for lang_kind in self.lang_kind_list])
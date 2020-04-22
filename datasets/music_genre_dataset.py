from torch.utils.data import Dataset
import os
from scipy.io import wavfile
import numpy as np
import torch

__all__ = ['MusicGenre_adv','MusicGenre','CLASSES2IDX']
CLASSES = 'blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock'.split(',')
CLASSES2IDX = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

def default_loader(path):
    sr, data = wavfile.read(path)
    # data = (2. / 65535.) * (data.astype(np.float32) - 32767) + 1.
    return data

class MusicGenre(Dataset):
    def __init__(self, data_folder, classes=None, default_loader=default_loader):
        file_list = os.listdir(data_folder)
        class_map = { c: i for c, i in zip(CLASSES,range(len(CLASSES)))}
        data = []
        for file in file_list:
            path = os.path.join(data_folder, file)
            for i in os.listdir(path):
                label = class_map[file]
                data.append((os.path.join(path,i), label))
            self.data = data
            self.default_loader = default_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        one_hot = np.zeros(10)
        one_hot[label] = 1
        audio = self.default_loader(path)
        audio = torch.FloatTensor(audio)
        one_hot = torch.FloatTensor(one_hot)
        return audio, label

class MusicGenre_adv(Dataset):
    def __init__(self, data_folder, target, classes=None, default_loader=default_loader):
        file_list = os.listdir(data_folder)
        class_map = { c: i for c, i in zip(CLASSES,range(len(CLASSES)))}
        data = []
        for file in file_list:
            if file == target:
                continue
            path = os.path.join(data_folder, file)
            for i in os.listdir(path):
                label = class_map[file]
                data.append((os.path.join(path,i), label))
            self.data = data
            self.default_loader = default_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        # one_hot = np.zeros(10)
        # one_hot[label] = 1
        audio = self.default_loader(path)
        audio = torch.FloatTensor(audio)
        # one_hot = torch.FloatTensor(one_hot)
        return audio, label
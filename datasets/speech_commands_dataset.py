import os
import numpy as np
from scipy.io import wavfile

import librosa
import torch
from transforms import *

from torch.utils.data import Dataset
from torchvision import datasets
__all__ = ['CLASSES','CLASSES_ALL', 'CLASSES2INDEX', 'SpeechCommandsDataset', 'BackgroundNoiseDataset', 'SpeechCommandsDataset_v2', 'BackgroundNoiseDataset_v2','SpeechCommandsDataset_classifier']

CLASSES = ['yes']  #.split(',') #,no,up,,down,left  right, on, off, stop, go
CLASSES_ALL = 'yes,no,up,down,left,right,on,off,stop,go'.split(',') #
CLASSES2INDEX = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9}


class SpeechCommandsDataset(Dataset):
    def __init__(self, folder, transform=None, classes=CLASSES, silence_percentage=0.1):

        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = 0
        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

            # add  silence
            # target = class_to_idx['silence']
            # data += [('', target)] * int(len(data) * silence_percentage)
            self.classes = classes
            self.data = data
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16384, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data

def minimaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max -min)

def default_loader(path, sample_rate=16384):
    # samples, sr = librosa.load(path, 16384)
    fn, wav_data = wavfile.read(path)
    if sample_rate < len(wav_data):
        wav_data = wav_data[:sample_rate]
    elif sample_rate > len(wav_data):
        wav_data = np.pad(wav_data, (0, sample_rate - len(wav_data)), "constant")
    wav_data = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    return wav_data

class SpeechCommandsDataset_v2(Dataset):
    def __init__(self, folder, target,transform=None, classes=CLASSES, loader=default_loader):
        # D input data
        all_data = CLASSES_ALL
        class_to_idx_data = {CLASSES_ALL[i]: i for i in range(len(CLASSES_ALL))}
        data_but_target = []
        for c in all_data:
            d = os.path.join(folder, c)
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data_but_target.append(path)
        # G input data
        all_classes = CLASSES

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            idx= CLASSES2INDEX[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, idx))


            # add  silence
        self.classes = classes
        self.data = data
        self.target_data = data_but_target
        self.loader = loader
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        np.random.shuffle(self.target_data)
        path, label = self.data[index]
        d_data_path = self.target_data[index]
        samples = self.loader(path)
        din_sample = self.loader(d_data_path)
        # data = {'samples': samples, 'label': target, 'sample_rate':sample_rate}
        g_in_data = torch.FloatTensor(samples)
        d_in_data = torch.FloatTensor(din_sample)
        return d_in_data, g_in_data, label


class BackgroundNoiseDataset_v2(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16384, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data


class SpeechCommandsDataset_classifier(Dataset):
    def __init__(self, folder,transform=None, classes=CLASSES, loader=default_loader):
       # all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_') ]
        all_classes = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')   #CLASSES
        # class_to_idx = {all_classes[i]: i for i in range(len(all_classes))}
        # for c in all_classes:
        #     if c not in class_to_idx:
        #         class_to_idx[c] = 0
        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = CLASSES2INDEX[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

            # add  silence
            self.classes = classes
            self.data = data
            self.loader = loader
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        samples = self.loader(path)
        # data = {'samples': samples, 'label': target, 'sample_rate':sample_rate}
        data = torch.FloatTensor(samples)
        return data, target
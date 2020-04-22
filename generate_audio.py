import torch
import torchvision
from models.generator import Generator
from scipy.io import wavfile
from datasets import *
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import numpy as np
import time

def wav_save(epoch, data, data_dir, label, target, name):
    singals = data.cpu().data.numpy()
    label = label.cpu().data.numpy()
    # classes2idx = {'unknown': 0, 'silence': 1, 'yes': 2, 'left': 3, 'right': 4}
    # idx2classes = {0: 'yes', 1: 'left', 2: 'right'}
    # idx2classes = {0: 'yes', 1: 'no', 2: 'up',3:'down',4:'left',5:'right',6:'on',7:'off',8:'stop',9:'go'}
    idx2classes = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',
                   8: 'reggae', 9: 'rock'}
    # for key, value in classes2idx.items():
    #     if value not in idx2classes:
    #         idx2classes[value] = key
    # print(idx2classes)
    for i in range(len(singals)):
        output = singals[i].reshape(16384, 1)
        # output = (output - 1) / (2 / 65535) + 32767
        # output = output.astype(np.int16)
        labels = idx2classes[label[i]]
        dir = os.path.join(data_dir, labels)
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        filename = "{}_{}_to_{}_epoch_{}_{}.wav".format(name, idx2classes[label[i]], idx2classes[target], epoch, i)
        path = os.path.join(dir, filename)
        wavfile.write(path, 16384, output)

def default_loader(path, sample_rate=16384):
    # samples, sr = librosa.load(path, 16384)
    fn, wav_data = wavfile.read(path)
    if sample_rate < len(wav_data):
        wav_data = wav_data[:sample_rate]
    elif sample_rate > len(wav_data):
        wav_data = np.pad(wav_data, (0, sample_rate - len(wav_data)), "constant")
    wav_data = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    return wav_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio_advGAN')
    parser.add_argument('--test_dataset', type=str, default='datasets/genres/test', help='')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--target', type=int, default=8)
    args = parser.parse_args()
    # {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8,
    #  'rock': 9}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    G = Generator().to(device)
    ckpt = torch.load(os.path.join(args.checkpoint, 'genres_classifiction/last-generator-checkpoint.pth'))  # last-generator-checkpoint
    G.load_state_dict(ckpt['state_dict'])
    test_dataset = MusicGenre(args.test_dataset)# valid_feature_transform
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    pbar = tqdm(test_loader, unit='generate audio', unit_scale=test_loader.batch_size)
    idx = 0
    num = 0
    time_sum = []
    for input, label in pbar:
        # if num >= 900:
        #     break
        start_time = time.time()
        input = torch.unsqueeze(input, 1)
        input = input.to(device)
        label = label.to(device)

        # gen = G(input)
        perturbation = torch.clamp(G(input), -0.3, 0.3)
        adv_audio = perturbation + input
        gen = adv_audio.clamp(-1., 1.)

        wav_save(idx, gen, 'generated/gen', label, args.target, 'fake')
        wav_save(idx, input, 'generated/real', label, args.target, 'real')
        end_time = time.time()
        epoch_time = end_time - start_time
        time_sum.append(epoch_time)
        print("Attack done in %0.4f seconds" % (end_time - start_time))
        idx += 1
        num += input.size(0)
    print('Total examples :', num)
    print('Total time :', sum(time_sum))
    print('Finished!')


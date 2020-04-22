from models.discriminator import Discriminator
from models.generator import Generator, Generator_pert
import torch
import numpy as np
import argparse
import os
from scipy.io import wavfile
from pytorch_mfcc import MFCC
import models
from torch.autograd import Variable

from datasets import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import *

def wav_save(epoch, data, data_dir, label, target, name):
    singals = data.cpu().data.numpy()
    label = label.cpu().data.numpy()
    # classes2idx = {'unknown': 0, 'silence': 1, 'yes': 2, 'left': 3, 'right': 4}
    idx2classes = {0: 'unknown', 1: 'silence', 2: 'yes', 3: 'left', 4: 'right'}
    # for key, value in classes2idx.items():
    #     if value not in idx2classes:
    #         idx2classes[value] = key
    # print(idx2classes)
    for i in range(len(singals)):
        output = singals[i].reshape(16384, 1)
        output = (output - 1) / (2 / 65535) + 32767
        output = output.astype(np.int16)
        labels = idx2classes[label[i]]
        dir = os.path.join(data_dir, labels)
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        filename = "{}_{}_to_{}_epoch_{}_{}.wav".format(name, idx2classes[label[i]], idx2classes[target], epoch, i)
        path = os.path.join(dir, filename)
        wavfile.write(path, 16384, output)
        # librosa.output.write_wav(path, output, 16384)

def test(epoch, target):
    pbar = tqdm(test_loader, unit="audios", unit_scale=test_loader.batch_size)
    for samples, labels in pbar:
        inputs = torch.unsqueeze(samples, 1)
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        fakes = G(inputs)
        # adv_audio = torch.clamp(pertubtion, -0.3, 0.3) + inputs
        # fakes = torch.clamp(adv_audio, -1.0, 1.0)
        wav_save(epoch, fakes, 'samples/gen', labels, target, 'fake')
        # wav_save(epoch, pertubtion, 'samples/pert', labels, target, 'pert')
        wav_save(epoch, inputs, 'samples/real', labels, target, 'real')


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
    parse = argparse.ArgumentParser(description="test pharse")
    parse.add_argument('--data_dir',type=str,default='datasets/speech_commands/test',help='the test data dir')
    parse.add_argument('--sample_dir',type=str,default='samples/fake')
    parse.add_argument('--checkpoint', type=str,default='checkpoints/')
    parse.add_argument('--task', choices=['speech_common_generator','music_genres_generator'], default='speech_common_generator', help='select the taks for attack')
    parse.add_argument('--generator', type=str, default='generator-checkpoint-epoch-target-yes.pth', help='the trained model')
    parse.add_argument('--input',type=str, default='', help='the speech to perturb')
    parse.add_argument('--batch_size',type=int,default=64)
    parse.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parse.add_argument('--model', choices=models.available_models, default=models.available_models[7],
                        help='model of NN(sampleCNN:0, wideResNet:7)')
    args = parse.parse_args()
    use_gpu = torch.cuda.is_available()

    G = Generator()
    f = models.create_model(model_name=args.model, num_classes=10, in_channels=1).cuda()
    mfcc_layer = MFCC(samplerate=16384, numcep=32, nfft=2048, nfilt=32).cuda()  # MFCC layer
    f = models.create_model(model_name=args.model, num_classes=10, in_channels=1)
    checkpoint = torch.load(
        'checkpoints/wideResNet28_10.pth')  # last-speech-commands-checkpoint_adv_wide50_classes
    # f.load_state_dict(checkpoint)  # sampleCNN
    f.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    f.eval()



    # test_dataset = SpeechCommandsDataset_v2(args.data_dir,
    #                                         transform=transforms.ToTensor())  # valid_feature_transform
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=use_gpu,
    #                          num_workers=args.dataload_workers_nums)

    if use_gpu:
        G = G.cuda()
        f.cuda()
    # the checkpoint path for different victim model
    checkpoint = os.path.join(args.checkpoint,args.task)

    if args.checkpoint:
        print("Loading a pretrained model ")
        checkpoint = torch.load(os.path.join(checkpoint, args.generator))
        G.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    # for epoch in range(5):
    #     test(epoch, 2)

    data_dir = r'D:\data\speech_commands_data\test1'
    # file = r'down\5f814c23_nohash_0.wav'
    # file = r'go\5ff3f9a1_nohash_0.wav'
    # file = r'left\4c841771_nohash_0.wav'
    # file = r'no\9a69672b_nohash_0.wav'
    # file = r'off\3efef882_nohash_0.wav'
    # file = r'on\2c6d3924_nohash_0.wav'
    # file = r'right\9a7c1f83_nohash_0.wav'
    # file = r'stop\1fe4c891_nohash_0.wav'
    file = r'up\f428ca69_nohash_0.wav'
    # file = r'yes\6f2f57c1_nohash_0.wav'
    output_path = 'blob/demo_output/result/yes/up'

    data = default_loader(os.path.join(data_dir,file))
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data,dim=0)
    data = torch.unsqueeze(data,dim=0).cuda()

    # label before perturb
    lengths = [16384]
    val, mfcc_lengths = mfcc_layer(torch.squeeze(data.detach(),dim=0), lengths)
    inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
    outputs = f(inputs)
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    pred = outputs.data.max(1, keepdim=True)[1]
    idx2classes = {0: 'yes', 1: 'no', 2: 'up', 3: 'down', 4: 'left', 5: 'right', 6: 'on', 7: 'off', 8: 'stop', 9: 'go'}
    print('the original label is:', idx2classes[pred.item()])

    perturbation = G(data)
    adv_audio = torch.clamp(perturbation, -0.3, 0.3) + data
    fake = torch.clamp(adv_audio, -1.0, 1.0)

    filename = os.path.basename(file)
    output = fake.cpu().data.numpy().reshape(16384, 1)
    output = (output - 1) / (2 / 65535) + 32767
    output = output.astype(np.int16)
    wavfile.write(os.path.join(output_path, filename), 16384, output)

    # prediction
    lengths = [16384]
    val, mfcc_lengths = mfcc_layer(torch.squeeze(fake.detach(),dim=1), lengths)
    inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
    outputs = f(inputs)
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    pred = outputs.data.max(1, keepdim=True)[1]
    idx2classes = {0: 'yes', 1: 'no', 2: 'up', 3: 'down', 4: 'left', 5:'right',6:'on',7:'off',8:'stop',9:'go'}
    print('the perturbed label is:', idx2classes[pred.item()])
    # statistic
    # D:\data\genres\test
    # blues\blues_00000_17.wav
    # classical\classical_00025_15.wav
    # country\country_00005_29.wav
    # disco\disco_00003_29.wav
    # hiphop\hiphop_00024_16.wav
    # jazz\jazz_00029_20.wav
    # metal\metal_00077_6.wav
    # pop\pop_00010_0.wav
    # reggae\reggae_00022_24.wav
    # rock\rock_00044_21.wav

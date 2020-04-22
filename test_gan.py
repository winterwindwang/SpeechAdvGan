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

def default_loader(path, sample_rate=16384):
    fn, wav_data = wavfile.read(path)
    if sample_rate < len(wav_data):
        wav_data = wav_data[:sample_rate]
    elif sample_rate > len(wav_data):
        wav_data = np.pad(wav_data, (0, sample_rate - len(wav_data)), "constant")
    wav_data = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    return wav_data

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="test pharse")
    parse.add_argument('--data_dir',type=str, required=True,help='the original speech to be perturbed')
    parse.add_argument('--task', choices=[0,1], default=0, help='0:speech_common_generator, 1:music_genres_generator')
    parse.add_argument('--target', type=str, required=True, default='', help='the trained model')
    parse.add_argument('--output_dir',type=str, default='./', help='the output dir of generated adversarial example')
    parse.add_argument('--checkpoint',type=str, required=True, default='checkpoints',help='the checkpoints')
    parse.add_argument('--model', choices=models.available_models, default=models.available_models[7],
                        help='model of NN(sampleCNN:0, wideResNet:7)')
    args = parse.parse_args()
    use_gpu = torch.cuda.is_available()

    G = Generator()
    if args.task == 0:
        f = models.create_model(model_name=args.model, num_classes=10, in_channels=1).cuda()
        mfcc_layer = MFCC(samplerate=16384, numcep=32, nfft=2048, nfilt=32).cuda()  # MFCC layer
        print("Loading a pretrained victim model ")
        checkpoint = torch.load(
            'checkpoints/wideResNet28_10.pth')  #
        f.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        f.eval()
        idx2classes = {0: 'yes', 1: 'no', 2: 'up', 3: 'down', 4: 'left', 5: 'right', 6: 'on', 7: 'off', 8: 'stop',
                       9: 'go'}
        generator_model_path = os.path.join(args.checkpoint, "speech_common_generator")
    else:
        f = models.create_model(model_name=args.model, num_classes=10, in_channels=1)
        print("Loading a pretrained victim model ")
        checkpoint = torch.load('checkpoints/sampleCNN.pth')  #
        f.load_state_dict(checkpoint)  # sampleCNN
        idx2classes = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal',
                       7: 'pop', 8: 'reggae', 9: 'rock'}
        generator_model_path = os.path.join(args.checkpoint, 'music_genres_generator')

    if use_gpu:
        G = G.cuda()
        f.cuda()

    if args.checkpoint:
        print("Loading a pretrained generator model ")
        generator_model = 'generator-checkpoint-epoch-target-' + args.target + '.pth'
        checkpoint = torch.load(os.path.join(generator_model_path, generator_model))
        G.load_state_dict(checkpoint['state_dict'])
        del checkpoint

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

    data = default_loader(args.data_dir)
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

    print('the original label is:', idx2classes[pred.item()])

    perturbation = G(data)
    adv_audio = torch.clamp(perturbation, -0.3, 0.3) + data
    fake = torch.clamp(adv_audio, -1.0, 1.0)

    (oname, extensrion) = os.path.split(os.path.basename(file))
    filename = 'fake_target_' + args.target +'_' + oname + extensrion
    output = fake.cpu().data.numpy().reshape(16384, 1)
    output = (output - 1) / (2 / 65535) + 32767
    output = output.astype(np.int16)
    wavfile.write(os.path.join(args.output_dir, filename), 16384, output)

    # prediction
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

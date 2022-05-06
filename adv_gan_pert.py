import argparse
import time
import os
from tqdm import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.io import wavfile
import torch.nn as nn
from torch import optim
from pytorch_mfcc import MFCC
import torchvision
from torchvision.transforms import *

from tensorboardX import SummaryWriter
import models
from models.discriminator import Discriminator
from models.generator import Generator, Generator_pert, Generator_speech
from datasets import *
# from transforms import *
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def wav_save(epoch, data, data_dir, label, target, name):
    singals = data.cpu().data.numpy()
    label = label.cpu().data.numpy()
    # idx2classes = {0: 'yes', 1: 'no', 2: 'up', 3: 'down', 4: 'left', 5:'right',6:'on',7:'off',8:'stop',9:'go'}
    idx2classes = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5:'jazz',6:'metal',7:'pop',8:'reggae',9:'rock'}

    for i in range(len(singals)):
        output = singals[i].reshape(16384, 1)
        # output = (output - 1) / (2 / 65535) + 32767
        # output = output.astype(np.int16)
        labels = idx2classes[label[i]]
        dir = os.path.join(data_dir, labels)
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = "{}_{}_to_{}_epoch_{}_{}.wav".format(name, idx2classes[label[i]], idx2classes[target], epoch, i)
        path = os.path.join(dir, filename)
        wavfile.write(path, 16384, output)


def train(epoch, target):
    global global_step
    f.eval()
    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase, get_lr(), epoch)
    acc = 0
    epoch_loss_g = 0
    epoch_loss_d = 0
    n = 0
    Tensor = torch.cuda.FloatTensor if "cuda" in device.__str__() else torch.FloatTensor
    pbar = tqdm(train_loader, unit='audios', unit_scale=train_loader.batch_size)
    for inputs, label in pbar:
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.to(device)
        label = label.to(device)

        G.zero_grad()
        perturbation = torch.clamp(G(inputs), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)

        if "wideresnet28_10" in args.model:
            lengths = [inputs.size(2) for _ in range(inputs.size(0))]
            val, mfcc_lengths = mfcc_layer(torch.squeeze(fakes), lengths)
            y_pred = f(torch.unsqueeze(val, dim=1))
        elif "sampleCNN" in args.model:
            y_pred = f(fakes)

        if args.is_targeted:
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            loss_adv = criterion_crossentropy(y_pred, y_target)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()

        else:
            y_true = Variable(label.to(device))
            loss_adv = criterion_crossentropy(y_pred, y_target)

            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()
        dg_fake,_ = D(fakes)
        g_loss_d = torch.mean((dg_fake - 1.0) ** 2)
        g_loss_l1 = torch.mean(torch.abs(torch.add(fakes, torch.neg(inputs))))

        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

        if epoch >= 3:
            g_loss = g_loss_d + loss_adv + 100 * g_loss_l1 + loss_perturb
        else:
            g_loss = g_loss_d + 100 * g_loss_l1 + loss_perturb
        g_loss.backward()
        g_optimizer.step()

        epoch_loss_g += g_loss.item()
        # Training Discriminator
        D.zero_grad()
        d_real, _ = D(inputs)
        d_loss_real = torch.mean((d_real - 1.0) ** 2)
        d_fake, _= D(fakes.detach())
        d_loss_fake = torch.mean(d_fake ** 2)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        d_optimizer.step()

        epoch_loss_d += d_loss.item()
        epoch_loss_d += 0
        global_step += 1
        n += inputs.size(0)
        # update the progress bar
        pbar.set_postfix({
            'd_loss': "%.05f" % (d_loss.mean().item()),
            'g_loss': "%.05f" % (g_loss.mean().item()),
            'loss_perturb': "%.05f" % (loss_perturb.mean().item()),
            'g_loss_gan': "%.05f" % (g_loss_d.mean().item()),
            'loss_adv': "%.05f" % (loss_adv.mean()),
            'acc': "%.02f" % (acc / n)
        })
    accuracy = acc / n
    epoch_loss_ds = epoch_loss_d / n
    epoch_loss_gs = epoch_loss_g / n
    writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
    writer.add_scalar('%s/epoch_loss_d' % phase, epoch_loss_ds, epoch)
    writer.add_scalar('%s/epoch_loss_g' % phase, epoch_loss_gs, epoch)


def valid(epoch, target):
    global best_accuracy, best_loss, global_step
    f.eval()
    G.eval()
    phase = 'valid'
    it = 0
    acc = 0

    n = 0
    epoch_loss_g = 0
    pbar = tqdm(valid_loader, unit="audios", unit_scale=valid_loader.batch_size)
    for samples, labels in pbar:
        inputs = torch.unsqueeze(samples, 1)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Valid Generator

        perturbation = torch.clamp(G(inputs), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)

        if "wideresnet28_10" in args.model:
            lengths = [inputs.size(2) for _ in range(inputs.size(0))]
            val, mfcc_lengths = mfcc_layer(torch.squeeze(fakes), lengths)
            y_pred = f(torch.unsqueeze(val, dim=1))
        elif "sampleCNN" in args.model:
            y_pred = f(fakes)

        if args.is_targeted:
            y_target = Variable(torch.ones_like(labels).fill_(target).to(device))
            loss_adv = criterion_crossentropy(y_pred, y_target)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = Variable(labels.to(device))
            loss_adv = criterion_crossentropy(y_pred, y_true)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()
        dg_fake, _ = D(fakes)
        g_loss_d = torch.mean((dg_fake - 1.0) ** 2)
        g_loss_l1 = torch.mean(torch.abs(torch.add(fakes, torch.neg(inputs))))
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        if epoch >= 3:
            g_loss = g_loss_d + loss_adv + 100 * g_loss_l1 + loss_perturb
        else:
            g_loss = g_loss_d + 100 * g_loss_l1 + loss_perturb

        epoch_loss_g += g_loss.item()

        # statistics
        n += labels.size(0)
        it += 1
        writer.add_scalar('%s/loss' % phase, g_loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'g_loss': "%.05f" % (g_loss.mean().item()),
            'acc': "%.02f" % (acc / n)
        })

    accuracy = acc / n
    epoch_loss = epoch_loss_g / it
    writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': G.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer': G.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # save_dir = os.path.join(args.wav_save_dir, 'best_attack')
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # wav_save(epoch, fakes, save_dir, labels, target, 'best_att')
        torch.save(checkpoint, f'{writer.logdir}/best-acc-generator-checkpoint-%s.pth' % full_name)
        torch.save(G, f'{writer.logdir}/%d-%s-best-loss.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, f'{writer.logdir}/best-loss-generator-checkpoint-%s.pth' % full_name)
        torch.save(G, f'{writer.logdir}/%d-%s-best-acc.pth' % (start_timestamp, full_name))

    # torch.save(checkpoint, f'{writer.logdir}/generator-checkpoint-epoch-%s.pth' % (epoch))
    torch.save(checkpoint, f'{writer.logdir}/last-generator-checkpoint.pth')
    del checkpoint  # reduce memory

    return epoch_loss


def test(epoch, target):
    pbar = tqdm(test_loader, unit="audios", unit_scale=test_loader.batch_size)
    for samples, labels in pbar:
        inputs = torch.unsqueeze(samples, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        perturbation = torch.clamp(G(inputs), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)

        wav_save(epoch, fakes, f'{args.wav_save_dir}/gen', labels, target, 'fake')
        wav_save(epoch, perturbation, f'{args.wav_save_dir}/pert', labels, target, 'pert')
        wav_save(epoch, inputs, f'{args.wav_save_dir}/real', labels, target, 'real')



if __name__ == '__main__':
    setup_seed(1022)
    parser = argparse.ArgumentParser(description='Audio_advGAN')
    parser.add_argument('--epochs', type=int, default=60, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--g_lr', type=float, default=1e-5, help='')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='')
    parser.add_argument('--train_dataset', type=str, default='dataset/train', help='datasets/speech_commands/train')
    parser.add_argument('--valid_dataset', type=str, default='dataset/valid', help='datasets/speech_commands/valid')
    parser.add_argument('--test_dataset', type=str, default='dataset/test', help='datasets/speech_commands/test')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='')
    parser.add_argument('--wav_save_dir', type=str, default='saved_wav', help='')
    parser.add_argument('--model', choices=models.available_models, default=models.available_models[1],
                        help='model of NN')
    parser.add_argument("--dataload-workers-nums", type=int, default=0, help='number of workers for dataloader')
    parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
    parser.add_argument("--pre_trained", type=bool, default=True, help='checkpoint file to resume')
    parser.add_argument("--max-epochs", type=int, default=50, help='max number of epochs')
    parser.add_argument("--optim", choices=['adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--is_targeted", type=bool, default=True, help='is target ')
    parser.add_argument("--target", type=str, default='yes', help='the target you wanted to attack')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Use GPU', device)

    G = Generator().to(device)
    D = Discriminator().to(device)
    mfcc_layer = MFCC(samplerate=16384, numcep=32, nfft=2048, nfilt=32).to(device)  # MFCC layer
    f = models.create_model(model_name=args.model, num_classes=10, in_channels=1).to(device)


    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    criterion_crossentropy = nn.CrossEntropyLoss()

    if "cuda" in device.__str__():
        torch.backends.cudnn.benchmarks = True
        f = torch.nn.DataParallel(f).cuda()
        criterion_gan.to(device)
        criterion_l1.to(device)
        criterion_l2.to(device)
        criterion_crossentropy.to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    # speech recognition
    if "wideresnet28_10" in args.model:
        train_dataset = SpeechCommandsDataset_v2(args.train_dataset, args.target)
        valid_dataset = SpeechCommandsDataset_v2(args.valid_dataset, args.target)
        test_dataset = SpeechCommandsDataset_v2(args.test_dataset, args.target)
        print("Loading a pretrained model ")
        checkpoint = torch.load(os.path.join(args.checkpoint, 'wideResNet28_10.pth'))
        f.load_state_dict(checkpoint['state_dict'])
        del checkpoint  # reduce memory
    elif "sampleCNN" in args.model:
        train_dataset = MusicGenre_adv(args.train_dataset,args.target)
        valid_dataset = MusicGenre_adv(args.valid_dataset,args.target)
        test_dataset = MusicGenre_adv(args.test_dataset,args.target)
        print("Loading a pretrained model ")
        checkpoint = torch.load(os.path.join(args.checkpoint, 'sampleCNN.pth'))
        f.load_state_dict(checkpoint)
        del checkpoint  # reduce memory

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.dataload_workers_nums, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.dataload_workers_nums, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=args.dataload_workers_nums, drop_last=True)

    start_timestamp = int(time.time() * 1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    def get_lr():
        return g_optimizer.param_groups[0]['lr']

    # a name used to save checkpoints etc.
    full_name = '%s_%s_bs%d_lr%.1e_wd%.1e' % (
        args.model, args.optim, args.batch_size, args.g_lr, args.weight_decay)
    writer = SummaryWriter(comment=('_speech_commands_' + full_name))
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        if "wideresnet28_10" in args.model:
            class_idx = CLASSES2INDEX[args.target]
        elif "sampleCNN" in args.model:
            class_idx = CLASSES2IDX[args.target]

        train(epoch, class_idx)
        epoch_loss = valid(epoch, class_idx)
        if epoch % 2 == 0:
            test(epoch, class_idx)

        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100 * best_accuracy, best_loss))
    print("finished")
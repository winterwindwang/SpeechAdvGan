#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

import argparse
import time
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision.transforms import *

from tensorboardX import SummaryWriter
from pytorch_mfcc import MFCC

import models
from datasets import *
# from transforms import *




def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase, get_lr(), epoch)
    mfcc_layer.eval()
    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for input, label in pbar:
        input = torch.unsqueeze(input, 1)


        lengths = [input.size(2) for _ in range(input.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)
        inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
        targets = Variable(label, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        # forward/backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        # print(loss.item())
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100 * correct / total)
        })

    accuracy = correct / total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)


def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for input, label in pbar:

        input = torch.unsqueeze(input, 1)
        lengths = [input.size(2) for _ in range(input.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)

        inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
        targets = Variable(label, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100 * correct / total)
        })

    accuracy = correct.item() / total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer': optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s_adv_wide50_classes.pth' % full_name)
        torch.save(model, 'runs/model/%d-%s-best-loss_adav_wide50_classes.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s_adv_wide50_classes.pth' % full_name)
        torch.save(model, 'runs/model/%d-%s-best-acc_adv_10_wide50_classes.pth' % (start_timestamp, full_name))

    torch.save(checkpoint, 'checkpoints/speechcommand/last-speech-commands-checkpoint_adv_wide50_classes.pth')
    del checkpoint  # reduce memory

    return epoch_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
    parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
    parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
    parser.add_argument("--batch-size", type=int, default=16, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
    parser.add_argument("--optim", choices=['sgd', 'adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
    parser.add_argument("--max-epochs", type=int, default=30, help='max number of epochs')
    parser.add_argument("--resume", type=str, help='checkpoint file to resume')
    parser.add_argument("--model", choices=models.available_models, default=models.available_models[8], help='model of NN')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True


    train_dataset = SpeechCommandsDataset_classifier(args.train_dataset, transform=transforms.ToTensor())
    valid_dataset = SpeechCommandsDataset_classifier(args.valid_dataset, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums,drop_last=True)

    # a name used to save checkpoints etc.
    full_name = '%s_%s_bs%d_lr%.1e_wd_adv_10_classes_mel40%.1e' % (args.model, args.optim, args.batch_size, args.learning_rate, args.weight_decay)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mfcc_layer = MFCC(samplerate=16384, numcep=32, nfft=2048, nfilt=32).to(device)  # MFCC layer
    model = models.create_model(model_name=args.model, num_classes=len(CLASSES_ALL), in_channels=1)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if args.resume:
        print("resuming a checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    def get_lr():
        return optimizer.param_groups[0]['lr']

    writer = SummaryWriter(comment=('_speech_commands_' + full_name))


    print("training %s for Google speech commands..." % args.model)
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        epoch_loss = valid(epoch)
        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
    print("finished")


import torch
import argparse
from models import sampleCNN
from datasets.music_genre_dataset import MusicGenre
import utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
import os

def eval(epoch):
    eval_loss = 0.0
    avg_auc1 = []
    avg_ap1 = []
    avg_auc2 = []
    avg_ap2 = []
    acc = 0
    n = 0
    phar = tqdm(valid_loader, unit='valid', unit_scale=valid_loader.batch_size)
    for audio, label in phar:
        audio = torch.unsqueeze(audio, dim=1)
        audio = audio.to(device)
        label = label.to(device)

        pred = sampleModel(audio)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        auc1, aprec1 = utils.tagwise_aroc_ap(label.cpu().data.numpy(), pred.cpu().data.numpy())
        avg_auc1.append(np.mean(auc1))
        avg_ap1.append(np.mean(aprec1))
        auc2, aprec2 = utils.itemwise_aroc_ap(label.cpu().data.numpy(), pred.cpu().data.numpy())
        avg_auc2.append(np.mean(auc2))
        avg_ap2.append(np.mean(aprec2))
        acc += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(label, dim=1))
        eval_loss += loss.item()
        n+=audio.size()[0]
    avg_loss = eval_loss / len(valid_loader)
    phar.set_postfix({
        'loss': avg_loss,
        'acc': acc.item() / n,
        'Retrieval: AROC': np.mean(avg_auc1),
        'Retrieval: AP ': np.mean(avg_ap1),
        'Annotation: AROC ': np.mean(avg_auc2),
        'Annotation: AP ': np.mean(avg_ap2),
    })
    # print("Retrieval : Average AROC = %.3f, AP = %.3f / " % (np.mean(avg_auc1), np.mean(avg_ap1)),
    #       "Annotation : Average AROC = %.3f, AP = %.3f" % (np.mean(avg_auc2), np.mean(avg_ap2)))
    # print('Average loss: {:.4f} \n'.format(avg_loss))
    return avg_loss

def train(epoch):
    avg_auc1 = []
    avg_ap1 = []
    avg_auc2 = []
    avg_ap2 = []
    i = 0
    n = 0
    acc = 0
    phar = tqdm(train_loader,unit='train', unit_scale=train_loader.batch_size)
    for audio, label in phar:
        audio = torch.unsqueeze(audio, dim=1)
        audio = audio.to(device)
        label = label.to(device)

        pred = sampleModel(audio)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # n += audio.size()[0].item()

        if (i + 1) % 10 == 0:
            # print("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (
            #     epoch + 1, args.epochs, i + 1, len(train_loader), loss.item()))

            # retrieval
            auc1, ap1 = utils.tagwise_aroc_ap(label.cpu().detach().numpy(), pred.cpu().detach().numpy())
            avg_auc1.append(np.mean(auc1))
            avg_ap1.append(np.mean(ap1))
            # annotation
            auc2, ap2 = utils.itemwise_aroc_ap(label.cpu().detach().numpy(), pred.cpu().detach().numpy())
            avg_auc2.append(np.mean(auc2))
            avg_ap2.append(np.mean(ap2))

            print("Retrieval : AROC = %.3f, AP = %.3f / " % (np.mean(auc1), np.mean(ap1)),
                  "Annotation : AROC = %.3f, AP = %.3f" % (np.mean(auc2), np.mean(ap2)))
        i += 1
        acc += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(label, dim=1))
        n+=audio.size()[0]
        phar.set_postfix({
            'loss': loss.item(),
            'acc': acc.item() / n,
            'Retrieval: AROC': np.mean(avg_auc1),
            'Retrieval: AP ': np.mean(avg_ap1),
            'Annotation: AROC ': np.mean(avg_auc2),
            'Annotation: AP ': np.mean(avg_ap2),
        })
        # print("Retrieval : Average AROC = %.3f, AP = %.3f / " % (np.mean(avg_auc1), np.mean(avg_ap1)),
        #       "Annotation :Average AROC = %.3f, AP = %.3f" % (np.mean(avg_auc2), np.mean(avg_ap2)))
        # print('Evaluating...')


    torch.save(sampleModel.state_dict(),
                'checkpoints/speechcommand/sampleCNN_' + str(epoch) + '.pth')





if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='CRNN')
    parse.add_argument('--train_dir', type=str, default='datasets/genres/train')
    parse.add_argument('--valid_dir', type=str, default='datasets/genres/valid')
    parse.add_argument('--test_dir', type=str, default='datasets/genres/test')
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--nb_classes', type=int, default=10)
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--lr', type=float, default=0.008)
    parse.add_argument('--model_savepath',type=str, default='./model')
    args = parse.parse_args()

    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    music_genre_train = MusicGenre(args.train_dir, classes=args.nb_classes)
    train_loader = DataLoader(music_genre_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    music_genre_valid= MusicGenre(args.valid_dir, classes=args.nb_classes)
    valid_loader = DataLoader(music_genre_valid, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    # (frequency, time, channel)
    x_shape = [args.batch_size, 32, 1]
    sampleModel = sampleCNN.SampleCNN01().to(device)
    optimizer = Adam(sampleModel.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
                                               verbose=True)
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(args.epochs):
        train(epoch)
        eval_loss = eval(epoch)

        scheduler.step(eval_loss)  # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-7:
            print("Early stopping")
            break
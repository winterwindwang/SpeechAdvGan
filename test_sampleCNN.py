import argparse
from datasets.music_genre_dataset import MusicGenre
import utils
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from models import sampleCNN


def test():
    avg_auc1 = []
    avg_ap1 = []
    avg_auc2 = []
    avg_ap2 = []
    acc = 0
    n = 0
    phar = tqdm(test_loader, unit='valid', unit_scale=test_loader.batch_size)
    for audio, label in phar:
        audio = audio.to(device)
        label = label.to(device)

        pred = sampleModel(audio)

        auc1, aprec1 = utils.tagwise_aroc_ap(label.cpu().data.numpy(), pred.cpu().data.numpy())
        avg_auc1.append(np.mean(auc1))
        avg_ap1.append(np.mean(aprec1))
        auc2, aprec2 = utils.itemwise_aroc_ap(label.cpu().data.numpy(), pred.cpu().data.numpy())
        avg_auc2.append(np.mean(auc2))
        avg_ap2.append(np.mean(aprec2))
        acc += torch.sum(torch.argmax(pred,dim=1)==torch.argmax(label, dim=1))
        n += audio.size()[0]
    phar.set_postfix({
        'acc': acc.item() / n,
        'Retrieval: AROC': np.mean(avg_auc1),
        'Retrieval: AP ': np.mean(avg_ap1),
        'Annotation: AROC ': np.mean(avg_auc2),
        'Annotation: AP ': np.mean(avg_ap2),
    })
    print('acc', acc.item() / n)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='CRNN')
    parse.add_argument('--train_dir', type=str, default=r'D:\data\genres\train')
    parse.add_argument('--valid_dir', type=str, default=r'D:\data\genres\valid')
    parse.add_argument('--test_dir', type=str, default=r'D:\data\genres\test')
    parse.add_argument('--batch_size', type=int, default=8)
    parse.add_argument('--nb_classes', type=int, default=10)
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--lr', type=float, default=0.008)
    parse.add_argument('--model_savepath',type=str, default='./model')
    args = parse.parse_args()

    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    music_genre_test = MusicGenre(args.test_dir, classes=args.nb_classes)
    test_loader = DataLoader(music_genre_test, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    sampleModel = sampleCNN.SampleCNN01().to(device)
    sampleModel.load_state_dict(torch.load("sampleCNN_49.pth"))
    test()   # acc acc 0.7758620689655172
    # valid 0.8524137931034482
import argparse
import models
from tqdm import *
from torch.utils.data import DataLoader
import torchnet.meter as meter
from pytorch_mfcc import MFCC

from datasets import *
from torchvision import transforms
import torch

def test():
    f.eval() # set model to evluate mode
    correct = 0
    total = 0
    confusion_matrix = meter.confusionmeter.ConfusionMeter(10)

    pbar = tqdm(test_loader, unit='audios', unit_scale=test_loader.batch_size)
    for input, label in pbar:
        inputs = torch.unsqueeze(input, 1)
        # lengths = [input.size(2) for _ in range(input.size(0))]
        # val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)
        # inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
        # targets = Variable(label, requires_grad=False)



        if "cuda" in device.__str__():
            inputs = inputs.cuda()
            targets = label.cuda(async=True)
        # forward

        if "wideresnet28_10" in args.model:
            lengths = [inputs.size(2) for _ in range(inputs.size(0))]
            val, mfcc_lengths = mfcc_layer(torch.squeeze(inputs), lengths)
            outputs = f(torch.unsqueeze(val, dim=1))
        elif "sampleCNN" in args.model:
            outputs = f(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        # statistic
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(torch.squeeze(pred), targets.data)

        accuacy = correct.item() / total
        print('accuracy: %f%%' %(100 * accuacy))
        print('confusion matrix:')
        print(confusion_matrix.value())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=64, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=3, help='number of workers for dataloader')
    parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument('--multi-crop', action='store_true', help='apply crop and average the results')
    parser.add_argument('--generate-kaggle-submission', action='store_true', help='generate kaggle submission file')
    parser.add_argument("--kaggle-dataset-dir", type=str, default='datasets/genres/test',
                        help='path of kaggle test dataset')
    parser.add_argument('--output', type=str, default='',
                        help='save output to file for the kaggle competition, if empty the model name will be used')
    parser.add_argument('--model', choices=models.available_models, default=models.available_models[1],
                        help='available_model: 1 for WideResNet trained on speech command dataset,'
                             '0 for SampleCNN trained on music genres dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='')
    parser.add_argument("--dataset-dir", type=str, default=r'D:\DataSource\speech_commands_v002\test', help='path of generated data')
    args = parser.parse_args()

    print('loading model...')
    f = models.create_model(model_name=args.model, num_classes=10, in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "cuda" in device.__str__():
        print('use_gpu', True)
        torch.backends.cudnn.benchmark = True
        f = torch.nn.DataParallel(f).to(device)  # If encouter the error of "Missing key(s) in state_dict, use this code can solve the error"
        # f.cuda()

    if "wideresnet28_10" in args.model:
        test_dataset = SpeechCommandsDataset_classifier(args.dataset_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.dataload_workers_nums)
        mfcc_layer = MFCC(samplerate=16384, numcep=32, nfft=2048, nfilt=32).to(device)  # MFCC layer

        checkpoint = torch.load('checkpoints/wideResNet28_10.pth')
        f.load_state_dict(checkpoint['state_dict'])

    elif "sampleCNN" in args.model:
        test_dataset = MusicGenre(args.dataset_dir)  # valid_feature_transform
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=args.dataload_workers_nums)
        checkpoint = torch.load(
            'checkpoints/sampleCNN.pth')  # last-speech-commands-checkpoint_adv_wide50_classes
        f.load_state_dict(checkpoint)  # sampleCNN
    # f.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    f.eval()
    pbar = tqdm(test_loader, unit='generate audio', unit_scale=test_loader.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    test()




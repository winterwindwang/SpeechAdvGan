import scipy.io.wavfile as wavfile
import numpy as np
import os
from glob import glob
from datasets import *

def wav_snr(ref_wav, in_wav):# 如果ref wav稍长，则用0填充in_wav
    if (abs(in_wav.shape[0] - ref_wav.shape[0]) < 10):
        pad_width = ref_wav.shape[0] - in_wav.shape[0]
        in_wav = np.pad(in_wav, (0, pad_width), 'constant')
    else:
        print("错误：参考wav与输入wav的长度明显不同")
        return -1

    # 计算 SNR
    norm_diff = np.square(np.linalg.norm(in_wav - ref_wav))
    if (norm_diff == 0.):
        print("错误：参考wav与输入wav相同")
        return -1

    ref_norm = np.square(np.linalg.norm(ref_wav))
    snr = 10 * np.log10(ref_norm / norm_diff)
    return snr


def processor(real_file, fake_file, folder):
    '''
    calculate the file .wav snr
    :param real_file:
    :param fake_file:
    :return:
    '''
    # real_list = os.listdir(real_file)
    # fake_list = os.listdir(fake_file)
    real_list = glob(os.path.join(real_file,'*.wav'))
    fake_list = glob(os.path.join(fake_file,'*.wav'))
    snr = []
    idx = 0
    for (i, j) in zip(real_list, fake_list):
        if idx == 125 and folder == 'hiphop':
            idx += 1
            continue
        fake_signal = wavfile.read(i)[1]
        real_signal = wavfile.read(j)[1]
        res = wav_snr(fake_signal, real_signal)
        snr.append(res)
        idx += 1
    return np.sum(snr) / len(real_list)

# fake = wavfile.read(r'C:\Users\wdh\Desktop\fake_up_to_yes_epoch_11_55.wav')[1]
# real = wavfile.read(r'C:\Users\wdh\Desktop\real_up_to_yes_epoch_11_55.wav')[1]
# print(wav_snr(fake,real))

dir_file_real = r'D:\王冬华\音频对抗样本\music_genres\攻击目标为rock\last generator\generated\real'
dir_file_gen = r'D:\王冬华\音频对抗样本\music_genres\攻击目标为rock\last generator\generated\gen'
# r'D:\王冬华\音频对抗样本\music_genres\攻击目标为rock\last generator\generated\real'
snr_sum = 0
# all_classes = 'yes,up,down,left,right,on,off,stop,go'.split(',')
all_classes = 'blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock'.split(',')
for i in range(len(all_classes)):
    snr = processor(os.path.join(dir_file_real,all_classes[i]),os.path.join(dir_file_gen,all_classes[i]), all_classes[i])
    print('snr for the class {} is {}'.format(all_classes[i], snr))
    snr_sum += snr
print(snr_sum / len(all_classes))
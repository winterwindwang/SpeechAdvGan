import librosa
import os
import shutil
import numpy as np

path = 'genres.tar/genres'
newFileFolder = 'data/genres'

file_list = os.listdir(path)
def resample_16k(path, file_list, newFileFolder):
    # file_dir = [os.path.join(path,i) for i in file_list if os.path.isdir(os.path.join(path,i))]
    for folder_name in file_list:
        # the file folder

        folder = os.path.join(path, folder_name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        for file in files:
            file_path = os.path.join(folder, file)
            y, sr = librosa.load(file_path)
            y_16k = librosa.resample(y, sr, 16384)
            print(len(y_16k)/16384)
            y_16k_list = [y_16k[i:i+16384] for i in range(0, len(y_16k)-1, 16384)]
            # the new folder to store new data
            newFolder = os.path.join(newFileFolder, folder_name)
            if not os.path.exists(newFolder):
                os.mkdir(newFolder)
            for i in range(len(y_16k_list)-1):
                newFilename = os.path.basename(file).split('.')[0] + '_' + os.path.basename(file).split('.')[1] + '_' + str(i) + '.wav'
                save_path = os.path.join(newFolder, newFilename)
                librosa.output.write_wav(save_path, y_16k_list[i], 16384)

def move_files(src_folder, to_folder, file_list):
    '''
    :param src_folder:source folder
    :param to_folder: destination folder
    :param file_list: file name like 'bed/0c40e715_nohash_0.wav'
    :return:
    '''
    for file in file_list:
        dirname = os.path.dirname(file)
        dest = os.path.join(to_folder, dirname)
        if not os.path.exists(dest):
            os.mkdir(dest)
        shutil.move(os.path.join(src_folder, file), dest)


def split_data(dir):
    valid_folder = os.path.join(dir, 'valid')
    test_folder = os.path.join(dir, 'test')
    train_folder = os.path.join(dir, 'train')
    if os.path.exists(valid_folder) is False:
        os.mkdir(valid_folder)
    if os.path.exists(test_folder) is False:
        os.mkdir(test_folder)
    dir = os.path.join(dir, 'audio')
    list_folder = os.listdir(dir)
    for folder in list_folder:
        file_list = os.listdir(os.path.join(dir, folder))
        index = np.random.permutation(len(file_list))
        dirname_list = [os.path.join(folder, file) for file in file_list]
        dirname_list = np.array(dirname_list)[index]
        move_files(dir, valid_folder, dirname_list[:290])
        move_files(dir, test_folder, dirname_list[-290:])
        # os.rename(dir, train_folder)
split_data('data/genres')
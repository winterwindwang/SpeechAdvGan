"""split the google speech commmands into train, validation and test data"""

import os
import shutil
import argparse


def move_files(src_folder, to_foler, list_file):
    with open(list_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            dirname = os.path.dirname(line)
            dest = os.path.join(to_foler, dirname)
            if not os.path.exists(dest):
                os.mkdir(dest)
            shutil.move(os.path.join(src_folder, line), dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split goodle commands train dataset')
    parser.add_argument("--root", type=str,default=r'D:\DataSource\speech_commands_v002', help='the path to the root folder of the google commands train dataset.')
    args = parser.parse_args()
    # audio_folder = os.path.join(args.root, 'train')
    audio_folder = os.path.join(args.root, 'speech_commands')

    print(audio_folder)
    validation_path = os.path.join(audio_folder, 'validation_list.txt')
    test_path = os.path.join(audio_folder, 'testing_list.txt')

    valid_folder = os.path.join(args.root, 'valid')
    test_folder = os.path.join(args.root, 'test')
    train_folder = os.path.join(args.root, 'train')

    if os.path.exists(valid_folder) is False:
        os.mkdir(valid_folder)
    if os.path.exists(test_folder) is False:
        os.mkdir(test_folder)
    # os.mkdir(test_folder)

    move_files(audio_folder, test_folder, test_path)
    move_files(audio_folder, valid_folder, validation_path)
    os.rename(audio_folder, train_folder)
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copied_model_rates
from copied_model_rates import main

DATA_COUNT = 1970


def read_all_data(path: str, classes: []):
    convert_tensor = transforms.ToTensor()
    list_of_classes, list_of_tens = [], []
    for clas in classes:
        for i, imgae in enumerate(os.scandir(path + '/' + str(clas))):
            try:
                list_of_tens.append(convert_tensor(Image.open(imgae.path)))
            except:
                print('My Dick is carrot')
        list_of_classes.extend([clas] * len(list((os.scandir(path + '/' + str(clas))))))

    return list_of_tens, list_of_classes


def pick_best_train_val(path_src: str, path_dest: str, classes: [], rate=10):
    best_acc = 0
    for i in range(10):
        copy_dirs('data/train_steve_improved', path_src)
        copy_dirs('data/val_steve_improved', path_dest)

        # make random dataset with given rate
        make_dummy_val_and_train(path_src, path_dest, rate, classes)

        _, _, _, acc = main(rate)

        if acc > best_acc:
            print('Best acc on a train-val set is', acc)
            copy_dirs(path_src, 'data/train_final')
            copy_dirs(path_src, 'data/val_final')


def determine_rates(path_src: str, path_dest: str, classes):
    rates = list(range(10, 35, 5))

    for rate in tqdm(rates):
        for i in range(5):
            create_dummy_dirs('data/train_steve_improved', path_src)
            create_dummy_dirs('data/val_steve_improved', path_dest)

            # make random dataset with given rate
            make_dummy_val_and_train(path_src, path_dest, rate, classes)

            main(rate)


def make_dummy_val_and_train(path_src: str, path_dest: str, rate: int, classes):

    for i in range(int((rate * DATA_COUNT) / 100)):
        clas = random.randint(0, 9)
        rand_file = random.choice(os.listdir(path_src + '/' + classes[clas]))
        shutil.move(path_src + '/' + classes[clas] + '/' + rand_file, path_dest + '/' + classes[clas] + '/' + rand_file)


def create_dummy_dirs(path_src: str, path_dest: str):

    if os.path.exists(path_dest):
        shutil.rmtree(path_dest)
    shutil.copytree(path_src, path_dest)


def copy_dirs(path_src: str, path_dest: str):

    if os.path.exists(path_dest):
        shutil.rmtree(path_dest)
    shutil.copytree(path_src, path_dest)

if __name__ == '__main__':
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    # list_of_tensors, labels = read_all_data('data/train', classes)
    # determine_rates('data/train_steve_dummy', 'data/val_steve_dummy', classes)
    pick_best_train_val('data/train_steve_dummyc', 'data/val_steve_dummyc', classes)
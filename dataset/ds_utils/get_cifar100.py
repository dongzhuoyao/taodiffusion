
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from scipy import misc
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

_root = "/home/thu/data_hhd"
meta = unpickle(f"{_root}/cifar-100-python/meta")

fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
with open("cifar100_fine_label_names.txt", "w") as f:
    f.write("\n".join(fine_label_names))

train = unpickle(f"{_root}/cifar-100-python/train")

filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
data = train[b'data']

images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)

with open('cifar100.csv', 'w+') as f:
    for index,image in tqdm(enumerate(images)):
        filename = filenames[index]
        label = fine_labels[index]
        label = fine_label_names[label]
        #misc.imsave('cifar100/img/train/%s' %filename, image)
        f.write('cifar100/img/train/%s,%s\n'%(filename,label))


# validation set example
#val_random = np.random.randint(0,len(train_set),5000)

#train_set.loc[val_random,'stage'] = 'VALIDATION'
#train_set.to_csv('data/cifar100.csv', index=False, header=False)

test = unpickle(f'{_root}/cifar-100-python/test')

filenames = [t.decode('utf8') for t in test[b'filenames']]
fine_labels = test[b'fine_labels']
data = test[b'data']

images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)

with open('cifar100.csv', 'a') as f:
    for index,image in tqdm(enumerate(images)):
        filename = filenames[index]
        label = fine_labels[index]
        label = fine_label_names[label]
        #misc.imsave('img/train/%s' %filename, image)
        f.write('cifar100/img/test/%s,%s\n'%(filename,label))

#print("Train examples: %i" %len(dataset[dataset['stage']=='TRAIN']))
#print("Validation examples: %i" %len(dataset[dataset['stage']=='VALIDATION']))
#print("Test examples: %i" %len(dataset[dataset['stage']=='TEST']))
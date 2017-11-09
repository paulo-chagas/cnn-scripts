import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt

from keras.utils import np_utils


def preprocess_input(x0):
    x = x0 / 255.
    #x -= 0.5
    #x *= 2.
    return x


def reverse_preprocess_input(x0):
    x = x0 / 2.0
    x += 0.5
    x *= 255.
    return x


def dataset(base_dir, n):
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            # if file_path.startswith(base_dir) is false then AssertionError
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())
    print 'Labels: ' + str(tags)

    processed_image_count = 0

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        # print(class_index, class_name)
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
            #img = scipy.misc.imread(filename)
            
            
            img = Image.open(filename)
            img = img.convert('RGB')
            img = np.asarray(img)


            #print 'shape: ' + str(img.shape)
            height, width, chan = img.shape
            assert chan == 3
            
            img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
            
            #plt.imshow(preprocess_input(img))
            #plt.show()
            X.append(img)
            y.append(class_index)

    print "Processed: %d images" % (processed_image_count)

    X = np.array(X).astype(np.float32)
    #print X.shape
    #X = X.transpose((0, 3, 1, 2))
    X = preprocess_input(X)
    y = np.array(y)

    # Shuffle data
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print "Classes:"
    for class_index, class_name in enumerate(tags):
        print class_name, sum(y==class_index)
    print 

    return X, y, tags


def main():
    train_data_dir = '../data_dir'
    #validation_data_dir = '../validation' #contains two classes cats and dogs
    in_prefix, n = train_data_dir, 299
    X, y, tags = dataset(in_prefix, n)
    #print X.shape
    nb_classes = len(tags)

    sample_count = len(y)
    X_train = X
    y_train = y
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    i=0
    for i in range(0,len(Y_train)):
        print y_train[i],Y_train[i] 


if __name__ == "__main__":
    main()
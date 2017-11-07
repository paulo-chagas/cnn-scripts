# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np
from datetime import datetime
import timeit
import os, errno

import dataset



log = []

# Avoid Tensorflow build warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Run Name
name = "../runs/inceptionV3_"
date_time = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
name += date_time
print name
log.append(name)

# Create path for images
path_img = name + "/imgs/"
log.append("path_img="+path_img)

# Create path for models
path_models = name + "/models/"
log.append("path_models="+path_models)

# Create path for TensorBoard logs
path_tb = name + "/logs_tensorboard/"
log.append("path_tb="+path_tb)

# Create folder if does not exist
try:
    os.makedirs(path_tb)
    os.makedirs(path_img)
    os.makedirs(path_models)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


csvLogger_transferLearning = CSVLogger(name+'/log_transferLearning.csv', 
                                        append=True, 
                                        separator=';')

csvLogger_fineTuning = CSVLogger(name+'/log_fineTuning.csv', 
                                        append=True, 
                                        separator=';')

tensorboard = TensorBoard(log_dir=path_tb,
                        histogram_freq=0, 
                        write_graph=True, 
                        write_images=True)

# dimensions of our images.
img_width, img_height = 299, 299
log.append("img_width, img_height = "+str(img_width) + ", " + str(299))

train_data_dir = '../data_dir'
log.append("train_data_dir="+train_data_dir)

validation_data_dir = '../validation'
log.append("validation_data_dir="+validation_data_dir)

nb_train_samples = 314
log.append("nb_train_samples = " + str(nb_train_samples))

nb_validation_samples = 40
log.append("nb_validation_samples = " + str(nb_validation_samples))

#nb_epoch = 50
nb_epoch = 10
log.append("nb_epoch = " + str(nb_epoch))

batch_size = 16
log.append("batch_size = " + str(batch_size))

# create the base pre-trained model
print "\n"
print "Creating model InceptionV3 --  no weights"
base_model = InceptionV3(weights=None, include_top=False)
weights_imagenet = '../weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
print "Loading weights..."
base_model.load_weights(weights_imagenet)
print "Done"

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#==========================================================================
print "Loading training data..."

X_train, y_train, tags = dataset.dataset(train_data_dir, img_width)
nb_classes = len(tags)

Y_train = np_utils.to_categorical(y_train, nb_classes)


print "Loading validation data..."

X_test, y_test, tags = dataset.dataset(validation_data_dir, img_width)

Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator()
datagen.fit(X_train)

#==========================================================================

def evaluate(model):
    log_evaluate = []
    confusion_csv = []
    Y_pred = model.predict(X_test)#, batch_size=16)
    y_pred = np.argmax(Y_pred, axis=1)

    accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    print "Accuracy:", accuracy
    log_evaluate.append("Accuracy:" +str(accuracy))
    
    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
        confusion[predicted_index, actual_index] += 1
    
    print "Rows are predicted classes, columns are actual classes\n"
    log_evaluate.append("Rows are predicted classes, columns are actual classes\n")
    print('\t\t'+'\t\t'.join(map(str,tags)))

    confusion_csv.append('-;'+';'.join(map(str,tags)))
    log_evaluate.append('\t\t'+'\t\t'.join(map(str,tags)))
    for predicted_index, predicted_tag in enumerate(tags):
        str_csv = predicted_tag[:]
        str_out = predicted_tag[:]
        print predicted_tag[:],
        for actual_index, actual_tag in enumerate(tags):
            str_csv += ";%d" % confusion[predicted_index, actual_index]
            str_out += "\t\t%d" % confusion[predicted_index, actual_index]
            print "\t\t%d" % confusion[predicted_index, actual_index],
        confusion_csv.append(str_csv)
        log_evaluate.append(str_out)
        print

    outs_eval = open(name+'/evaluation.txt', 'w+')
    csv_eval = open(name+'/evaluation.csv', 'w+')

    for item in log_evaluate:
        outs_eval.write("%s\n" % item)
    for item in confusion_csv:
        csv_eval.write("%s\n" % item)

#==========================================================================

print "Start history model - training the top layers"
start_time = timeit.default_timer()

history = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
    epochs=nb_epoch,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
    callbacks = [tensorboard, csvLogger_transferLearning],
    validation_steps=nb_validation_samples//batch_size) #1020

elapsed = timeit.default_timer() - start_time
print "Training the top layers: " + str(elapsed) + " seconds"
log.append("Training the top layers: " + str(elapsed) + " seconds")

print "Done\n"
print "Saving model"
model.save(path_models+"inception_retrained_top_layers")
print "Model saved - top layers"

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Acurácia - Treino Top Layers')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig(path_img+'result_acc_toplayers.png', bbox_inches='tight')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss - Treino Top Layers')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig(path_img+'result_loss_toplayers.png', bbox_inches='tight')


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
print "Layers\n"
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
learningR = 0.0001
decayR = 1e-6
momentum_ = 0.9
nesterov_ = True

model.compile(optimizer=SGD(lr=learningR, decay=decayR, momentum=momentum_, nesterov=nesterov_), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

log.append("optimizer=SGD(lr="+str(learningR)+", decay="+ str(decayR)+", momentum="+str(momentum_)+", nesterov=" + str(nesterov_))

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
# fine-tune the model

print "Fine-Tuning the model..."

start_time = timeit.default_timer()

history = model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            epochs=nb_epoch,
            steps_per_epoch=nb_train_samples//batch_size,
            validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
            validation_steps=nb_validation_samples//batch_size,
            callbacks=[csvLogger_fineTuning, tensorboard])

elapsed = timeit.default_timer() - start_time
print "Fine-Tuning: " + str(elapsed) + " seconds"
log.append("Fine-Tuning: " + str(elapsed) + " seconds")

print "Done\n"
print "Saving model"
model.save(path_models+"inception_retrained_model")
print "Model saved"

plt.clf()
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Acurácia - Fine-Tuning')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig(path_img+'result_acc_finetuning.png', bbox_inches='tight')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss - Fine-Tuning')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig(path_img+'result_loss_finetuning.png', bbox_inches='tight')

# save log to file
outs = open(name+'/outs.txt', 'w+')

for item in log:
    outs.write("%s\n" % item)

# Evaluate model
evaluate(model)

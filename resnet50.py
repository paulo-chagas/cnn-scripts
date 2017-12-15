# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger, Callback
import matplotlib.pyplot as plt
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils

import numpy as np
from datetime import datetime
import timeit
import os, errno

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import dataset



class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.accs.append(logs.get('acc'))

log = []

# Avoid Tensorflow build warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Run Name
name = "../runs/resnet/resnet_"
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
img_width, img_height = 224, 224
log.append("img_width, img_height = "+str(img_width) + ", " + str(img_height))

train_data_dir = '../base_train/chart_png_dir'
log.append("train_data_dir="+train_data_dir)

validation_data_dir = '../base_validation/chart_png_dir'
log.append("validation_data_dir="+validation_data_dir)

base_real_validation_dir = '../base_real_validation/'
log.append("base_real_validation="+base_real_validation_dir)

#nb_epoch = 50
nb_epoch = 25
log.append("nb_epoch = " + str(nb_epoch))

nb_epoch_top = 10
log.append("nb_epoch_top = " + str(nb_epoch_top))

batch_size = 16
log.append("batch_size = " + str(batch_size))

nb_train_samples = 4860
log.append("nb_train_samples = " + str(nb_train_samples))

nb_validation_samples = 972
log.append("nb_validation_samples = " + str(nb_validation_samples))

nb_validationReal_samples = 1301
log.append("nb_validation_samples = " + str(nb_validationReal_samples))


# create the base pre-trained model
print "\n"
print "Creating model ResNet50 --  no weights"
base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
weights_imagenet = '../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
print "Loading weights..."
base_model.load_weights(weights_imagenet)
print "Done"

# build a classifier model to put on top of the convolutional model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional resnet layers
for layer in base_model.layers:
    layer.trainable = False


print "Layers\n"
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)


lr = 1e-5
log.append("learning rate for training the top layers = " + str(lr))
# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

#==========================================================================
print "Loading training and validation data..."

train_datagen = ImageDataGenerator(
        )#rescale=1./255)#,
 #       shear_range=0.2,
 #       zoom_range=0.2,
 #       horizontal_flip=True)

test_datagen = ImageDataGenerator()#rescale=1./255)

validation_datagen = ImageDataGenerator()#rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)




#==========================================================================

def evaluate(model, eval_dir, eval_name, samples):
    print "Evaluation - " + eval_dir + " - " + eval_name
    validationReal_datagen = ImageDataGenerator()#rescale=1./255)

    validation_real_generator = validationReal_datagen.flow_from_directory(
        eval_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    #Confution Matrix and Classification Report
    validation_real_generator.reset()
    Y_pred = model.predict_generator(validation_real_generator, (samples//batch_size)+1)
    y_pred = np.argmax(Y_pred, axis=1)
    #for i,j,k in zip(validation_real_generator.filenames,validation_real_generator.classes, y_pred):
    #    print ((i,j),k)
    print('Confusion Matrix')
    confusionM = confusion_matrix(validation_real_generator.classes, y_pred)
    accuracyEval = accuracy_score(validation_real_generator.classes, y_pred) 
    print(confusionM)
    print "Accuracy " + str(accuracyEval)
    print('Classification Report')
    target_names = ['Bar', 'Line', 'Pie', 'Scatter']
    report = classification_report(validation_real_generator.classes, y_pred, target_names=target_names)
    print(report)

    outs_eval = open(name+'/evaluation_'+eval_name+'.txt', 'w+') 
    outs_eval.write("Evaluation on %s samples\nAccuracy: %s\nConfusion Matrix\n%s\n\n%s" % (samples, accuracyEval, confusionM,report))



    '''outs_eval = open(name+'/evaluation.txt', 'w+')
    csv_eval = open(name+'/evaluation.csv', 'w+')

    for item in log_evaluate:
        outs_eval.write("%s\n" % item)
    for item in confusion_csv:
        csv_eval.write("%s\n" % item)'''

#==========================================================================

print "Start history model - training the top layers"

history1 = AccHistory()

start_time = timeit.default_timer()

history = model.fit_generator(
    train_generator,
    epochs=nb_epoch_top,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    validation_data=validation_generator,
    callbacks = [history1, tensorboard, csvLogger_transferLearning],
    validation_steps=nb_validation_samples//batch_size) #1020

elapsed = timeit.default_timer() - start_time
print "Training the top layers: " + str(elapsed) + " seconds"
log.append("Training the top layers: " + str(elapsed) + " seconds")
total = elapsed

print "Done\n"
print "Saving model"
model.save(path_models+"resnet50_retrained_top_layers")
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
# convolutional layers from resnet. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
'''
print "Layers\n"
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
'''

# we will freeze the first 17 layers and unfreeze the rest:
for layer in model.layers[:141]:
   layer.trainable = False
for layer in model.layers[141:]:
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

# we train our model again (this time fine-tuning the top 1 inception blocks

#x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#x = AveragePooling2D((7, 7), name='avg_pool')(x)

# alongside the top Dense layers
#model.fit_generator(...)
# fine-tune the model

print "Fine-Tuning the model..."

history2 = AccHistory()

start_time = timeit.default_timer()

history = model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch=(nb_train_samples//batch_size)+1,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size,
            callbacks=[history2, csvLogger_fineTuning, tensorboard])

elapsed = timeit.default_timer() - start_time
print "Fine-Tuning: " + str(elapsed) + " seconds"
log.append("Fine-Tuning: " + str(elapsed) + " seconds")
total += elapsed
total = (total/60)/60
log.append("Tempo Total: " + str(total) + " hrs")
print "Tempo Total: " + str(total) + " hrs"

print "Done\n"
print "Saving model"
model.save(path_models+"resnet50_retrained_model")
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

outs2 = open(name+'/transferl-accs-val_accs.csv', 'w+')
for item in history1.accs:
    outs2.write("%s\n" % item)

outs3 = open(name+'/finet-accs-val_accs.csv', 'w+')
for item in history2.accs:
    outs3.write("%s\n" % item)

# Evaluate model
evaluate(model, base_real_validation_dir, 'base_real', nb_validationReal_samples)
evaluate(model, validation_data_dir, 'base_sintetica', nb_validation_samples)



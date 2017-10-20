# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

print "Acurácia"
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

import time
import os, errno

# Create path for TensorBoard logs
path = "../logs_tb/tb_{}".format(time.time())
print path

# Create folder if does not exist
try:
    os.makedirs(path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


tensorboard = TensorBoard(log_dir=path,
                        histogram_freq=0, 
                        write_graph=True, 
                        write_images=True)

# dimensions of our images.
img_width, img_height = 299, 299

train_data_dir = '../data_dir' #contains two classes cats and dogs
validation_data_dir = '../validation' #contains two classes cats and dogs

nb_train_samples = 314
nb_validation_samples = 40
nb_epoch = 50

# create the base pre-trained model
print "\n\n\n\n"
print "Creating model InceptionV3 --  no weights"
base_model = InceptionV3(weights=None, include_top=False)
weights_imagenet = '../inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
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

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
         rescale=1./255)#,
 #       shear_range=0.2,
 #       zoom_range=0.2,
 #       horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

print "Start history model - training the top layers"
history = model.fit_generator(
    train_generator,
    epochs=nb_epoch,
    steps_per_epoch=(nb_train_samples/16),
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples/16)) #1020

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Acurácia - Treino Top Layers')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig('../imgs/result_acc_toplayers_1.png', bbox_inches='tight')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss - Treino Top Layers')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig('../imgs/result_loss_toplayers_1.png', bbox_inches='tight')



# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
print "Layers"
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
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
# fine-tune the model

print "Fine-Tuning the model..."
history = model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch=(nb_train_samples/16),
            validation_data=validation_generator,
            validation_steps=(nb_validation_samples/16),
            callbacks=[tensorboard])

print "Done"
print "Saving model"
model.save("../models/inception_retrained_model_1")
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

plt.savefig('../imgs/result_acc_finetuning_1.png', bbox_inches='tight')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss - Fine-Tuning')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')

plt.savefig('../imgs/result_loss_finetuning_1.png', bbox_inches='tight')

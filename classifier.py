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

from datetime import datetime
import timeit
import os, errno

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
log.append("img_width, img_height = "+str(img_width) + ", " + str(299)

train_data_dir = '../data_dir'
log.append("train_data_dir="+train_data_dir)

validation_data_dir = '../validation'
log.append("validation_data_dir="+validation_data_dir)

nb_train_samples = 314
log.append("nb_train_samples = " + nb_train_samples)

nb_validation_samples = 40
log.append("nb_validation_samples = " + nb_validation_samples)

#nb_epoch = 50
nb_epoch = 5
log.append("nb_epoch = " + nb_epoch)

batch_size = 16
log.append("batch_size = " + batch_size)

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

# prepare data augmentation configuration
train_datagen = ImageDataGenerator()
 #       rescale=1./255)#,
 #       shear_range=0.2,
 #       zoom_range=0.2,
 #       horizontal_flip=True)

test_datagen = ImageDataGenerator()#rescale=1./255)

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

print "Start history model - training the top layers"
start_time = timeit.default_timer()

history = model.fit_generator(
    train_generator,
    epochs=nb_epoch,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_data=validation_generator,
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
lr = 


model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

log.append("optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True")

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
# fine-tune the model

print "Fine-Tuning the model..."

start_time = timeit.default_timer()

history = model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch=nb_train_samples//batch_size,
            validation_data=validation_generator,
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

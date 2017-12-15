from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



import numpy as np

import dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print "Loading model"
model = load_model('../runs/vgg/VGG_28-11-2017__19-52-03/models/vgg19_retrained_top_layers')
model.load_weights('../runs/vgg/VGG_28-11-2017__19-52-03/models/vgg19_retrained_top_layers')
print "Model loaded"
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print "Compile done" 


#==========================================================================

img_width, img_height = 224, 224

base_real_validation_dir = '../base_real_validation/'

nb_validationReal_samples = 1301

batch_size = 16


print "Loading validation data..."

validationReal_datagen = ImageDataGenerator()#rescale=1./255)

validation_real_generator = validationReal_datagen.flow_from_directory(
    base_real_validation_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    shuffle = False,
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



evaluate(model, base_real_validation_dir, 'base_real', nb_validationReal_samples)

# predicting images
'''
img0 = image.load_img('../base_real_validation/line/line382.jpg', target_size=(img_width, img_height))

x0 = image.img_to_array(img0)
x0 = np.expand_dims(x0, axis=0)
preds = model.predict(x0)
print preds
classes = np.argmax(preds)
print classes
'''
"""
x0 = preprocess_input(x0)
print x0.shape

img1 = image.load_img('bars_2.jpg', target_size=(img_width, img_height))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
print x1.shape

img2 = image.load_img('../002.jpg', target_size=(img_width, img_height))
x2 = image.img_to_array(img2)
x2 = np.expand_dims(x2, axis=0)
x2 = preprocess_input(x2)
print x2.shape

img3 = image.load_img('../003.jpg', target_size=(img_width, img_height))
x3 = image.img_to_array(img3)
x3 = np.expand_dims(x3, axis=0)
x3 = preprocess_input(x3)
print x3.shape

#images = np.vstack([x0, x1, x2, x3])
print "Prediction"
for i in [x0, x1, x2, x3]:
	preds = model.predict(i)
	print preds
	classes = np.argmax(preds)
	print labels[classes]
	#print('Predicted:', decode_predictions(preds, top=3)[0])
	print "------------"

"""
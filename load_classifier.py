from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.utils import np_utils

import numpy as np

import dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


img_width, img_height = 299, 299

labels = ['bar', 'bubble', 'donut', 'line']

print "Loading model"
model = load_model('../runs/inceptionV3_31-10-2017__17-31-45/models/inception_retrained_model')
model.load_weights('../runs/inceptionV3_31-10-2017__17-31-45/models/inception_retrained_model')
print "Model loaded"
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print "Compile done" 


#==========================================================================

img_width, img_height = 299, 299
validation_data_dir = '../validation'


print "Loading validation data..."

X_test, y_test, tags = dataset.dataset(validation_data_dir, img_width)
nb_classes = len(tags)

Y_test = np_utils.to_categorical(y_test, nb_classes)


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

    outs_eval = open('evaluation.txt', 'w+')
    csv_eval = open('evaluation.csv', 'w+')

    for item in log_evaluate:
        outs_eval.write("%s\n" % item)
    for item in confusion_csv:
        csv_eval.write("%s\n" % item)

evaluate(model)

# predicting images
"""
img0 = image.load_img('bars_1.jpg', target_size=(img_width, img_height))
x0 = image.img_to_array(img0)
x0 = np.expand_dims(x0, axis=0)
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
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

img_width, img_height = 299, 299

labels = ['bar', 'bubble', 'donut', 'line']

print "Loading model"
model = load_model('../models/inception_retrained_model_1')
model.load_weights('../models/inception_retrained_model_1')
print "Model loaded"
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print "Compile done" 

# predicting images
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
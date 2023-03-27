import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#######################################################
FOLDER = 'Kvasir-SEG'
FILE = 'cju306x7w05nb0835cunv799x.jpg'
MODEL_NAME = 'test.h5'
#######################################################

path_img = './datasets/{}/images/{}'.format(FOLDER,FILE)
path_mask = './datasets/{}/masks/{}'.format(FOLDER,FILE)

for i in [path_img,path_mask]:
    img = cv2.imread(i)
    cv2.imshow(i,img)

# LOAD MODEL
model = tf.keras.models.load_model('./models/{}'.format(MODEL_NAME),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})

array = get_files([path_img],type_of_file='image')
prediction = model.predict(array)

binario = np.where(prediction > 0.5, 1, 0)

plt.imshow(binario[0], interpolation='nearest')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

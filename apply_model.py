import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#######################################################
FOLDER = 'TEST_IMAGES'
FILE = '133_cvc.jpg'
MODEL_NAME = 'model_dice_loss_v1_b4'
#######################################################

model_name = '{}.h5'.format(MODEL_NAME)
history_name = 'history/{}.csv'.format(MODEL_NAME)

path_img = './datasets/{}/images/{}'.format(FOLDER,FILE)
path_mask = './datasets/{}/masks/{}'.format(FOLDER,FILE)

# for i in [path_img,path_mask]:
    # img = cv2.imread(i)
    # cv2.imshow(i,img)

# LOAD MODEL
model = tf.keras.models.load_model('./models/{}'.format(model_name),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})

array = get_files([path_img],type_of_file='image')
prediction = model.predict(array)

binario = np.where(prediction > 0.5, 1, 0)
kernel = np.ones((5,5),np.uint8)
binario = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel)

plt.imshow(binario[0], interpolation='nearest')
plt.show()

# history_table = pd.read_csv(history_name)
# history_class = GetHistory(history_table)
# history_class.accuracy_vs_val_accuracy()
# history_class.loss_vs_val_loss()

cv2.waitKey(0)
cv2.destroyAllWindows()

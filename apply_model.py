import cv2
from complement_functions import *
import tensorflow as tf

paths = ['./datasets/CVC-ClinicDB/images/248.jpg']
model_name = 'threshold_dice_loss.h5'


# LOAD MODEL
model = tf.keras.models.load_model('./models/{}'.format(model_name))


# for path in paths:
#     img = cv2.imread(path)
#     cv2.imshow(path,img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# array = get_files(paths,type_of_file='image')
# prediction = model.predict(array)

# print(prediction)

# binario = np.where(prediction > 0.5, 1, 0) * 255.0

# print(binario)

# from matplotlib import pyplot as plt
# plt.imshow(binario[0], interpolation='nearest')
# plt.show()

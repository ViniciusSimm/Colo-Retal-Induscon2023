import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class EvaluateModel:
    def __init__(self, FOLDER, FILES, MODEL_NAME):
        self.FOLDER = FOLDER
        self.FILES = FILES
        self.MODEL_NAME = MODEL_NAME

    def load_model(self):
        model = tf.keras.models.load_model('./models/{}.h5'.format(self.MODEL_NAME),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})
        return model
    
    def save_history(self):
        history_name = 'history/{}.csv'.format(self.MODEL_NAME)
        history_table = pd.read_csv(history_name)
        history_class = GetHistory(history_table)
        acc = history_class.accuracy_vs_val_accuracy()
        loss = history_class.loss_vs_val_loss()

        acc.savefig("output_midia/{}_accuracy.png".format(self.MODEL_NAME))
        loss.savefig("output_midia/{}_loss.png".format(self.MODEL_NAME))
        print('history')
    
    def get_paths(self):
        paths = []
        paths_mask = [] 
        for FILE in self.FILES:
            path_img = './datasets/{}/images/{}'.format(self.FOLDER,FILE)
            path_mask = './datasets/{}/masks/{}'.format(self.FOLDER,FILE)
            paths.append(path_img)
            paths_mask.append(path_mask)
        
        return paths, paths_mask
    
    def save_masks(self):
        paths, paths_mask = self.get_paths()
        model = self.load_model()

        for path in paths:
            array = get_files([path],type_of_file='image')
            prediction = model.predict(array)
            binario = np.where(prediction > 0.5, 1, 0)
            plt.imshow(binario[0], interpolation='nearest')
            plt.savefig("output_midia/{}_{}.png".format(self.MODEL_NAME,re.search(r'[\\/]([\w-]+)\.jpg', path).group(1)))

    def get_performance(self):
        paths, paths_mask = self.get_paths()
        model = self.load_model()
        array = get_files(paths,type_of_file='image')
        array_mask = get_files(paths_mask,type_of_file='mask') 
        prediction = model.predict(array)
        binario = np.where(prediction > 0.5, 1, 0)
        arrays_mask = tf.cast(array_mask, tf.int32)
        arrays_predict = tf.cast(binario, tf.int32)     
        if len(arrays_mask.shape) < 4:
            arrays_mask = tf.expand_dims(arrays_mask, axis=-1)  # [batch_size, height, width, 1]
            arrays_mask = tf.repeat(arrays_mask, repeats=1, axis=-1)  # [batch_size, height, width, channels]
        ssim = score_ssim(arrays_predict,arrays_mask)
        # print('SSIM:', ssim)
        mse = score_mse(arrays_predict,arrays_mask)
        # print('MSE:', mse)
        jaccard = score_jaccard(arrays_predict,arrays_mask)
        # print('JACCARD:', jaccard)
        euclidean = score_euclidean(arrays_predict,arrays_mask)
        # print('EUCLIDEAN:', euclidean)
        precision,recall = confusion_matrix(arrays_predict,arrays_mask)
        # print('PRECISION:', precision)
        # print('RECALL:', recall)

        with open("output_midia/metrics.txt", "a") as f:
            f.write("MODEL: {}\n".format(self.MODEL_NAME))
            f.write("FILES: {}\n".format(self.FILES))
            f.write("ssim: {}\n".format(ssim))
            f.write("jaccard: {}\n".format(jaccard))
            f.write("euclidean: {}\n".format(euclidean))
            f.write("precision: {}\n".format(precision))
            f.write("recall: {}\n\n".format(recall))


if __name__ == "__main__":
    print('inicio')
    FOLDER = 'TEST_IMAGES'
    FILES = ['133_cvc.jpg','419_cvc.jpg','480_cvc.jpg','cju1cfhyg48bb0799cl5pr2jh_kev.jpg','cju2hqt33lmra0988fr5ijv8j_kev.jpg']
    MODEL_NAME = 'UNET_no_preprocess_with_crop_v3'
    evaluatemodel = EvaluateModel(FOLDER, FILES, MODEL_NAME)
    evaluatemodel.save_history()
    evaluatemodel.save_masks()
    evaluatemodel.get_performance()
    print('fim')
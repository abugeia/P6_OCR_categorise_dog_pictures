import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from os import environ

############################
# Remove warnings
############################
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()

############################
# Parser with 2 arguments
############################
parser = argparse.ArgumentParser(description='Classify dog breed from image')
parser.add_argument('--path', help='path of the image file to be classified ', required=True)
parser.add_argument('--pic_show', help='if specified show the picture ', default=False, action="store_true")                  

args = parser.parse_args()
path = args.path
pic_show = args.pic_show

############################
# Model and labels loading
############################
model_path = './model/'
model = load_model(model_path + 'vgg16_fine_tuning_whit_cropped.h5', compile = False)

open_file = open(model_path + 'labels', "rb")
labels = pickle.load(open_file)
open_file.close()

############################
# Prediction section
############################
def predict(path, model, pic_show):
    '''predict the breeed of the picture'''
    img = keras.preprocessing.image.load_img(
        path, target_size=(224, 224)
    )
    if pic_show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)

    for pourc, name in zip(predictions[0], labels) :
        print("{:.2f}%".format(pourc*100), "\t", name)

predict(path, model, pic_show)
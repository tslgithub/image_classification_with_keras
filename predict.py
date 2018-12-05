import sys
import cv2
import os
import numpy as np

from config import config
from  VGG16 import vgg16
from keras.preprocessing.image import img_to_array


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config1 = tf.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config1))

class Predict(object):

    def __init__(self,config):
        self.classes = config.classes
        self.normal_size = config.normal_size
        self.checkpoints = config.checkpoints
        self.model_name = config.model_name
        self.channels = config.channels
        self.test_data_path = "./dataset/test/"+sys.argv[1]+"/"
        self.lr = config.lr

    def build_model(self):
        model = vgg16(normal_size=self.normal_size, channles=self.channels, classes=self.classes,lr=self.lr).VGG_16()
        return model

    def predict(self):
        # data_list = os.listdir(self.test_data_path)
        model = self.build_model()
        model.load_weights(self.checkpoints+self.model_name+'.h5')
        i, j, tmp = 0, 0, []
        data_list = os.listdir(self.test_data_path)
        for file in data_list:
            file_name = os.path.join(self.test_data_path,file)
            if self.channels == 1:
                img = cv2.imread(file_name,0)
            else:
                img = cv2.imread(file_name)
            img = cv2.resize(img,(self.normal_size,self.normal_size))
            img = img_to_array(img)

            data = np.array([img],dtype='float')/255.0
            pred = model.predict(data)
            pred = pred.tolist()
            label =  pred[0].index(max(pred[0]))
            print('predict:',label)

            if int(label) != int(sys.argv[1]):
                print('wrong label :_____________________________________________',label)
                i+=1
                tmp.append(label)
            else:
                j+=1

        print('error number: ', i, '\ntotal: ', i + j, '\naccuacy is: ', 1.0 - i / (i + j))
        print('error: ', ','.join(list(map(lambda x: str(x), tmp))))
        print('Done')

def main():
    predicts = Predict(config)
    predicts.predict()

if __name__=='__main__':
    main()
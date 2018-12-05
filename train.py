from  VGG16 import vgg16
from config  import config
import os
from itertools import chain
import glob
import tqdm
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import json

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

class Train(object):

    def __init__(self,config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints #'./checkpints/'
        self.normal_size = config.normal_size
        self.channels = config.channels
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.classes = config.classes
        self.data_agumentation = config.data_agumentation
        self.model_name = config.model_name
        self.lr = config.lr


    def get_file(self,catgories_path):
        if len((glob.glob(catgories_path+'/'+'*.png'))) >0:
            return glob.glob(catgories_path+'/'+'*.png')
        elif len((glob.glob(catgories_path+'/'+'*.jpg'))) >0:
            return glob.glob(catgories_path + '/' + '*.jpg')

    def load_data(self):
        labels ,images_data = [],[]
        categories_path = list(map(lambda x:self.train_data_path+x,os.listdir(self.train_data_path)))
        print(categories_path)
        files = list(chain.from_iterable(map(self.get_file,categories_path)))
        for file in tqdm.tqdm(files):
            label = file.split('/')[-2]
            labels.append(label)
            if int(self.channels)==1:
                image = cv2.imread(file,0)
            else:
                image = cv2.imread(file)
            image = cv2.resize(image,(self.normal_size,self.normal_size))
            image = img_to_array(image)
            images_data.append(image)

        images_data = np.array(images_data,dtype='float')/255.0
        labels = np.array(labels)
        labels = to_categorical(labels,num_classes=self.classes)
        return images_data,labels

    def build_model(self):
        model = vgg16(normal_size=self.normal_size,channles=self.channels,classes=self.classes,lr=self.lr).VGG_16()
        return model

    def train(self,X_train, X_test, y_train, y_test,model):
        csv_logger = CSVLogger('training.log',append=False)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                      patience=5, min_lr=1e-12,verbose=1)

        checkpoint= keras.callbacks.ModelCheckpoint(self.checkpoints + self.model_name + '.h5',
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='max', period=1)

        early_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=30,verbose=1,mode='max')

        if self.data_agumentation:
            print('using data agumentation')
            data_agu = ImageDataGenerator(rotation_range=5,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2
                                           )
            data_agu.fit(X_train)
            hist = model.fit_generator(data_agu.flow(x=X_train, y=y_train, batch_size=self.batch_size),
                                epochs=self.epochs,
                                verbose=1,
                                steps_per_epoch=X_train.shape[0] // self.batch_size,
                                callbacks=[reduce_lr, checkpoint, csv_logger, early_stop],
                                validation_data=(X_test, y_test),
                                shuffle=True)
        else:
            hist = model.fit(x=X_train, y=y_train, batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1,

                      callbacks=[reduce_lr,checkpoint,csv_logger,early_stop],
                      validation_data=(X_test,y_test),
                      shuffle=True)

        # with open('./train.log','a') as f:
        #     js = json.dumps(hist.history)
        #     f.write(js)
        print(hist.history)

    def start_train(self):
        images_data, labels = self.load_data()#load data
        X_train, X_test, y_train, y_test = train_test_split(images_data, labels)#data split
        model = self.build_model()#build model
        self.train(X_train, X_test, y_train, y_test,model)#start to train model

    # def

def main():
    g1 = tf.Graph()
    with g1.as_default():
        config1 = tf.ConfigProto()
        config1.gpu_options.per_process_gpu_memory_fraction = 0.4
        set_session(tf.Session(config=config1))
        train=Train(config)
        train.start_train()

if __name__=='__main__':
    main()
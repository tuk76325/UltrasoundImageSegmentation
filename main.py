import tensorflow as tf #Needed to get access to convolutional neural network model
import keras #used to train u-net model
import random #used to ensure that each image and mask file were lined up
import os #used to navigate through the file path and get access to all the images
import cv2 #used to process the mask and image files into multidimensional arrays to be read by the model
import numpy as np #used for data manipulation
import pandas as pd #used for data manipulation
from PIL import Image #used to resize and square the image files
from resizeimage import resizeimage #used to resize
from skimage.io import imread, imshow #used to read images for processing
import matplotlib.pyplot as plt #used to visualize data

#resizing images while keeping their aspect ratio, adding white bars to ensure the images are square
def make_square(im, min_size=128, fill_color=(200, 200, 200, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resizer():
    f = r'Test'
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = make_square(img)
        img = img.resize((128,128))
        img.save(f_img)

def modelBuild():
    trainPath = 'Train'
    testPath = 'Test'

    trainMasks = []
    trainImages = []
    testImages = []

    #Creates a pandas dataframe where all of the mask and image files are appended
    for i in os.listdir(trainPath):
        if 'mask' in i:
            trainImages.append('Train\\' + i.replace('_mask',''))
            trainMasks.append(os.path.join(trainPath, i))

    df = pd.DataFrame({'image': trainImages, 'mask': trainMasks})

    for i in os.listdir(testPath):
        testImages.append(os.path.join(testPath, i))

    testDF = pd.DataFrame({'image' : testImages})

    #Chooses at layer and mask at random and overlays them to ensure that the dataset is lined up
    # randInt = random.randint(0, len(df))
    # image_path = df['image'][randInt]
    # mask_path = df['mask'][randInt]
    #
    # print(image_path)
    # print(mask_path)
    # image1 = np.array(Image.open(image_path))
    # image1_mask = np.array(Image.open(mask_path))
    # image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)
    #
    # fig, ax = plt.subplots(1,3,figsize = (16,12))
    # ax[0].imshow(image1, cmap = 'gray')
    #
    # ax[1].imshow(image1_mask, cmap = 'gray')
    #
    # ax[2].imshow(image1, cmap = 'gray', interpolation = 'none')
    # ax[2].imshow(image1_mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)
    # plt.show()

    #Building the Model starting with convolutional neural network
    imgWidth = 128
    imgHeight = 128
    imgChannels = 3

    Xtrain = np.zeros((len(trainImages), imgHeight, imgWidth, imgChannels), dtype=np.uint8) #Image training dataset
    Ytrain = np.zeros((len(trainMasks), imgHeight, imgWidth, 1), dtype=bool) #Mask training dataset
    Xtest = np.zeros((len(trainImages), imgHeight, imgWidth, imgChannels), dtype=np.uint8) #Image test dataset

    #Converts images/masks to numpy array (128,128,3) to be read into the model
    for i in range(len(df)):
        image = cv2.imread(df['image'][i])[:,:,:imgChannels]
        Xtrain[i] = image
        mask_ = np.zeros((imgHeight, imgWidth, 1), dtype=bool)
        mask = cv2.imread(df['mask'][i])
        mask = np.expand_dims(mask.resize((128, 128)), axis=-1)  #convert from (128,128,3) ---> (128,128,1
        Ytrain[i] = mask

    for i in range(len(testDF)):
        image = cv2.imread(testDF['image'][i])[:,:,:imgChannels]
        Xtest[i] = image

    print(Xtrain.shape)
    print(Ytrain.shape)


    inputs = tf.keras.layers.Input(shape=(imgWidth, imgHeight, imgChannels))

    s = ks.layers.Lambda(lambda x: x/255)(inputs) #convert integer to floats so that the model can process the inputs

    # Downscaling path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Upscaling pathway
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(Xtrain, Ytrain, batch_size=32, validation_split=0.1, epochs=10)
    print('\n')
    preds_test = model.predict(Xtest, verbose=1)
    print('\n')
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

modelBuild()

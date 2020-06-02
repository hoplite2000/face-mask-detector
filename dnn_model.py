from tensorflow import keras
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
import h5py

Ini_lr = 1e-4
Epochs = 20
Bsize = 32

#pre process image and labels
def img_preprocess(dataset):
    print("[INFO] loading images...")
    imgpaths = list(paths.list_images(dataset))
    data = []
    labels = []

    # preprocess image to be fed into mobilenet v2
    for imgpath in imgpaths:
        label = imgpath.split(os.path.sep)[-2]
        img = keras.preprocessing.image.load_img(imgpath, target_size=(224, 224))
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.applications.mobilenet_v2.preprocess_input(img)

        labels.append(label)
        data.append(img)

    # converted to numpy arrays
    labels = np.array(labels)
    data = np.array(data, dtype='float32')

    # one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = keras.utils.to_categorical(labels)

    return data, labels

#prepare dataset
def prepare_dataset(data, labels):
    # prepare training and testing dataset
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    return trainX, testX, trainY, testY

#model architecture
def get_model():
    input_layer = keras.layers.Input(shape=(224, 224, 3))
    basemodel = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False,
                                                            input_tensor=input_layer)

    x = basemodel.output
    x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.models.Model(inputs=basemodel.input, outputs=x)

    for layer in basemodel.layers:
        layer.trainable = False

    return model

#main
path = './dataset'
data, labels = img_preprocess(path)
trainX, testX, trainY, testY = prepare_dataset(data, labels)
model = get_model()
opt = keras.optimizers.Adam(lr=Ini_lr, decay=Ini_lr/Epochs)
model.compile(optimizer = opt, loss = "binary_crossentropy", metrics=['accuracy'])

#augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                             zoom_range=0.15,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             shear_range=0.15,
                                                             horizontal_flip=True,
                                                             fill_mode="nearest")

train_generator = train_datagen.flow(trainX, trainY, batch_size=Bsize)

#train the network
history = model.fit(train_generator, steps_per_epoch=len(trainX)//Bsize,
                    validation_data=(testX, testY), validation_steps=len(testX)//Bsize,
                    epochs=Epochs)

#save the model
if not os.path.exists('./model'):
    os.makedirs('./model')
model.save('./model/model.model', save_format="h5")
model.save('./model/model.h5', save_format="h5")
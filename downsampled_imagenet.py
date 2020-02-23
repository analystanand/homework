import tensorflow as tf
from   tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import math
from keras.regularizers import l2
import numpy as np

IMAGE_SHAPE = (224, 224)
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


IMAGE_RES_mobilenet = 224
TRAINING_BATCH_SIZE = 32
TESTING_BATCH_SIZE = 32

IMAGE_ROWS = 224
IMAGE_COLS = 224
CHANNELS = 3

NUM_CLASSES = 1001
TRAINING_LR_MAX = 0.001

dataset_train, info = tfds.load("downsampled_imagenet/64x64",split="train", with_info=True)

dataset_test = tfds.load("downsampled_imagenet/64x64", split='test')

# get labels for dataset_train
train_batches = dataset_train.batch(TRAINING_BATCH_SIZE)

label_train = []

for train_batch in train_batches:
    image_batch = train_batch['image'].numpy()

    image_batch = tf.image.resize(image_batch,(224, 224),)

    predicted_batch = classifier.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()

    predict_labels = np.argmax(predicted_batch, axis=1)
    label_train.extend(predict_labels)

temp_d = tf.data.Dataset.from_tensor_slices(label_train)
train_labels = tf.data.Dataset.zip((dataset_train, temp_d))


test_batches = dataset_test.batch(TESTING_BATCH_SIZE)
label_test = []


for  test_batch in test_batches:

  image_batch = test_batch['image'].numpy()

  image_batch = tf.image.resize(image_batch,(224, 224),)

  predicted_batch = classifier.predict(image_batch)
  predicted_batch = tf.squeeze(predicted_batch).numpy()

  predict_labels = np.argmax(predicted_batch, axis=1)
  label_test.extend(predict_labels)

temp_test_d = tf.data.Dataset.from_tensor_slices(label_test)
test_labels = tf.data.Dataset.zip((dataset_test, temp_test_d))




def create_model(level_0_repeats, level_1_repeats, level_2_repeats):
    # encoder - input
    model_input = keras.Input(shape=(IMAGE_ROWS, IMAGE_COLS, CHANNELS), name='input_image')
    x = model_input

    # encoder - level 0
    for n0 in range(level_0_repeats):
        x = keras.layers.Conv2D(32, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # encoder - level 1
    for n1 in range(level_1_repeats):
        x = keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # encoder - level 2
    for n2 in range(level_2_repeats):
        x = keras.layers.Conv2D(128, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)

    # encoder - output
    encoder_output = x

    # decoder
    y = keras.layers.GlobalAveragePooling2D()(encoder_output)
    decoder_output = keras.layers.Dense(NUM_CLASSES, activation='softmax')(y)

    # forward path
    model = keras.Model(inputs=model_input, outputs=decoder_output, name='cifar_model')

    # loss, backward path (implicit) and weight update
    model.compile(optimizer=tf.keras.optimizers.Adam(TRAINING_LR_MAX), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


LEVEL_0_BLOCKS = 8
LEVEL_1_BLOCKS = 10
LEVEL_2_BLOCKS = 3
model = create_model(LEVEL_0_BLOCKS, LEVEL_1_BLOCKS, LEVEL_2_BLOCKS)

model.summary()

print(model.metrics_names)


def change_size(img, label):
  img = tf.image.resize(
      img['image'],
      (224, 224),
  )
  return img, label
EPOCHS = 500
itr1 = train_labels.batch(TRAINING_BATCH_SIZE).map(change_size).prefetch(2)
print(train_labels)
model.fit(itr1, steps_per_epoch=EPOCHS, verbose=1)


def change_size(img, label):
  img = tf.image.resize(
      img['image'],
      (224, 224),
  )
  return img, label

EPOCHS = 500
itr1 = label_test.batch(TRAINING_BATCH_SIZE).map(change_size).prefetch(2)

test_loss, test_accuracy = model.evaluate(itr1)

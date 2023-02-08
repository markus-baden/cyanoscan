import pathlib
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RSEED = 42

NUM_LAYERS = 4 #8
IMAGE_SIZE = 128
IMG_SHAPE = (NUM_LAYERS, IMAGE_SIZE, IMAGE_SIZE)
NUM_TOTAL_VALUES = NUM_LAYERS * IMAGE_SIZE * IMAGE_SIZE
DTYPE = tf.float32

# We set some parameters for the model
HEIGHT = IMAGE_SIZE #image height
WIDTH = IMAGE_SIZE #image width
CHANNELS = NUM_LAYERS #image channels (RGB+IR, RGB+IR masked)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 10 * BATCH_SIZE
AUTOTUNE = tf.data.experimental.AUTOTUNE

VALIDATION_SIZE = 1705
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE


# Define the function that decodes in the images
def decode_image(image, reshape_dim):
    # JPEG is a compressed image format. So we want to 
    # convert this format to a numpy array we can compute with.
    #image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # 'decode_jpeg' returns a tensor of type uint8. We need for 
    # the model 32bit floats. Actually we want them to be in 
    # the [0,1] interval.
    #image = tf.image.convert_image_dtype(image, tf.float32)
    # Now we can resize to the desired size.
    #image = tf.image.resize(image, reshape_dim)
    
    #image = tf.io.decode_raw(image, DTYPE)
    image = tf.reshape(image, reshape_dim,)
    return image

# The train set actually gives only the paths to the training images.
# We want to create a dataset of training images, so we need a 
# function that can handle this for us.
def decode_dataset(data_row):
    record_defaults = ['path', tf.constant([0.0], dtype=tf.float32)]
    filename, target = tf.io.decode_csv(data_row, record_defaults)
    
    # dataset = tf.data.FixedLengthRecordDataset(
    #     filenames=[filename],
    #     record_bytes=NUM_TOTAL_VALUES * DTYPE.size, 
    #     header_bytes=128
    # )
    # image_bytes = dataset.as_numpy_iterator().next()


    #image_bytes = tf.io.read_file(filename=filename)
    # remove header bytes # (32 bytes for float32, 16 bytes for float64, 64 bytes for float16)
    #image_bytes = repr(image_bytes)[32:]

    #np_image_layers = np.load(filename)
    #image_tensor = tf.convert_to_tensor(np_image_layers, np.float32)
    #label = tf.math.equal(label_string, CLASS_NAMES)
    
    #return image_bytes, target
    return filename, target

# Next we construct a function for pre-processing the images.
def read_and_preprocess(image_bytes, target, augment_randomly=False):
    if augment_randomly: 
        # TODO: Use augmentation here.
        # image = decode_image(image_bytes, [HEIGHT + 8, WIDTH + 8])
        # TODO: Augment the image.
        # image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])
        image = decode_image(image_bytes, IMG_SHAPE)
    else:
        # image = decode_image(image_bytes, [HEIGHT, WIDTH])
        image = decode_image(image_bytes, IMG_SHAPE)
    return image, target

def read_and_preprocess_with_augmentation(image_bytes, target): 
    return read_and_preprocess(image_bytes, target, augment_randomly=True)


# Load the numpy files
def numpy_loader(filename):
    # use all layers
    #image_array = np.load(str(filename)[2:-1])
    # only use masked layers
    image_array = np.load(str(filename)[2:-1])[4:]
    return image_array

def wrapper_func(filename, target):
    img = tf.numpy_function(numpy_loader, [filename], tf.float32)
    return img, target

# Now we can create the dataset.
def load_dataset(file_of_filenames, batch_size, training=True):
    # We create a TensorFlow Dataset from the list of files.
    # This dataset does not load the data into memory, but instead
    # pulls batches one after another.
    dataset = tf.data.TextLineDataset(filenames=file_of_filenames)
    dataset = dataset.map(decode_dataset)

    dataset = dataset.map(wrapper_func)#, num_parallel_calls=tf.data.AUTOTUNE)
    
    # dataset = dataset.map(lambda img, target: tf.numpy_function(
    #       numpy_loader, [img, target], tf.float32),
    #       num_parallel_calls=tf.data.AUTOTUNE)
    
    # if training:
    #     dataset = dataset.map(read_and_preprocess_with_augmentation).\
    #         shuffle(SHUFFLE_BUFFER).\
    #         repeat(count=None) # Infinite iterations
    # else: 
    #     # Evaluation or testing
    #     dataset = dataset.map(read_and_preprocess).\
    #         repeat(count=1) # One iteration

    dataset = dataset.map(read_and_preprocess).repeat(count=1) # One iteration

            
    # The dataset will produce batches of BATCH_SIZE and will
    # automatically prepare an optimized number of batches while the prior one is
    # trained on.
    return dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

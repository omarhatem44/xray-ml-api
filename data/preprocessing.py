import tensorflow as tf

IMG_SIZE = (224, 224)

def preprocess_image(img):
    """
    Shared preprocessing for training & inference
    """
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

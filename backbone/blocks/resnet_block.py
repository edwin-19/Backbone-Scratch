import tensorflow as tf

def identity_block(x, filters, kernel_size):
    f1, f2, f3 = filters
    
    x_shortcut = x
    
    # First block of identity
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second block of identity
    x = tf.keras.layers.Conv2D(f2, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Third block of identity 
    x = tf.keras.layers.Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    
    # Skip Conncection
    x = tf.keras.layers.Add()([x, x_shortcut])
    
    # Activaton to fireoff the next layer
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def convolutional_block(x, filters, kernel_size, strides):
    f1, f2, f3 = filters
    
    x_shortcut = x
    
    # First block of convolutional block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=strides)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second block of convolutional block
    x = tf.keras.layers.Conv2D(f2, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Third block of convolutional block
    x = tf.keras.layers.Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    
    # Shortcut path
    x_shortcut = tf.keras.layers.Conv2D(f3, kernel_size=(1, 1), strides=strides, padding='valid')(x_shortcut)
    x_shortcut = tf.keras.layers.BatchNormalization(axis=3)(x_shortcut)
    
    # Skip connection
    x = tf.keras.layers.Add()([x, x_shortcut])
    
    # Fire off to the last layer
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_first_block(input_shape=(224, 224, 3)):
    input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_layer)
    
    # Stage 1
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    return x, input_layer
    
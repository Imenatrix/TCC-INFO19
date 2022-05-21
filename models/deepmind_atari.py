from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense

def create_model(input_shape, nb_outputs):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = Conv2D(64, 4, strides=4, activation='relu')(x)
    #x = Conv2D(64, 3, strides=1, activation='relu')(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    output = Dense(nb_outputs, activation='linear')(x)

    return Model(inputs=inputs, outputs=output)
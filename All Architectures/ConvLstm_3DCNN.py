import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model_3dcnn_convlstm(input_shape=(9, 372, 743, 1)):
    inputs = layers.Input(shape=input_shape)  # (T, H, W, C)

    # === Extraction locale spatio-temporelle avec 3D CNN
    x = layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),  # (time, height, width)
        padding='same',
        activation='relu'
    )(inputs)  # (9, 372, 743, 32)
    x = layers.BatchNormalization()(x)

    # === temporal sequence with ConvLSTM2D
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        activation='tanh'
    )(x)  
    x = layers.BatchNormalization()(x)

    # === Prédiction NDVI final (1 canal)
    outputs = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation='sigmoid',  # NDVI normalisation between 0 and 1
        padding='same'
    )(x)  # (372, 743, 1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    model = build_model_3dcnn_convlstm(input_shape=(12, 372, 743, 1))
    model.summary()

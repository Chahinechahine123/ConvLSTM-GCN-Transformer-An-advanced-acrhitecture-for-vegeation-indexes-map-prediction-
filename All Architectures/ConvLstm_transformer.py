import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model_convlstm_transformer(input_shape=(9, 372, 743, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # === Spatio-temporal feature extraction via ConvLSTM
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        activation='tanh'
    )(inputs)
    x = layers.BatchNormalization()(x)  # Shape: (372, 743, 64)

    # === Transformer-like block
    h, w, c = x.shape[1], x.shape[2], x.shape[3]  # 372, 743, 64
    x = layers.Reshape((h * w, c))(x)  # (372*743, 64)
    
    positions = tf.range(start=0, limit=h * w, delta=1)
    pos_encoding = layers.Embedding(input_dim=h * w, output_dim=c)(positions)
    x = x + pos_encoding

    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # === Reconstruction
    x = layers.Reshape((h, w, 64))(x)
    outputs = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Test the model
if __name__ == "__main__":
    model = build_model_convlstm_transformer((9, 372, 743, 1))
    model.summary()

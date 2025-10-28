import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Dropout
from tensorflow.keras.models import Sequential

def build_model_convlstm(input_shape):
    """
    input_shape = (seq_len, H, W, 1)
    sortie      = (H, W, 1)  normalisée [0,1]
    """
    model = Sequential([
        ConvLSTM2D(64, (3,3), padding="same", return_sequences=True,
                   input_shape=input_shape),
        BatchNormalization(),

        ConvLSTM2D(32, (3,3), padding="same", return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Conv2D(1, (3,3), activation="sigmoid", padding="same")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Exemple de résumé
if __name__ == "__main__":
    mdl = build_model_convlstm((9, 743, 372, 1))
    mdl.summary(line_length=120)

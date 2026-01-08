import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# === 1. Dimensions ===
H, W = 372, 743
N = H * W

# === 2. Matrice d’adjacence (8-neighborhood) ===
def create_grid_adjacency_sparse_8n(h, w):
    row_idx = []
    col_idx = []

    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in neighbors_8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    j = rr * w + cc
                    row_idx.append(i)
                    col_idx.append(j)

    indices = np.stack([row_idx, col_idx], axis=1)
    values = np.ones(len(row_idx), dtype=np.float32)

    A_sparse = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[N, N]
    )

    return tf.sparse.reorder(A_sparse)

A_sparse = create_grid_adjacency_sparse_8n(H, W)

# === 3. Couche GCN ===
class GraphConvLayer(layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):
        F_in = input_shape[-1]
        self.W = self.add_weight(
            name="W_gcn",
            shape=(F_in, self.out_features),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="b_gcn",
            shape=(self.out_features,),
            initializer="zeros",
            trainable=True
        )

    def call(self, X):  # (batch, N, F)
        def graph_conv_single(Xb):
            AXb = tf.sparse.sparse_dense_matmul(A_sparse, Xb)
            return tf.matmul(AXb, self.W) + self.b

        return tf.map_fn(graph_conv_single, X)

# === 4. Modèle ConvLSTM + GCN ===
def build_model_convlstm_gnn(input_shape=(9, H, W, 1)):
    inputs = layers.Input(shape=input_shape)

    # ConvLSTM
    x = layers.ConvLSTM2D(
        64, (3, 3),
        padding="same",
        return_sequences=False
    )(inputs)
    x = layers.BatchNormalization()(x)

    # Image → Graphe
    x = layers.Reshape((N, 64))(x)

    # GCN (8-neighbors)
    x = GraphConvLayer(32)(x)
    x = layers.Activation("relu")(x)

    # Graphe → Image
    x = layers.Reshape((H, W, 32))(x)

    # Sortie
    outputs = layers.Conv2D(
        1, (3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model

# === 5. Test ===
if __name__ == "__main__":
    model = build_model_convlstm_gnn()
    model.summary()

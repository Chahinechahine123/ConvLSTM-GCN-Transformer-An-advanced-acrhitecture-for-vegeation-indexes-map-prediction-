import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# === 1. Dimensions ===
H, W = 372, 743  
N = H * W

# === 2. Sparse adjency matrix
def create_grid_adjacency_sparse(h, w):
    row_idx = []
    col_idx = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighboor
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    j = rr * w + cc
                    row_idx.append(i)
                    col_idx.append(j)
    indices = np.stack([row_idx, col_idx], axis=1)
    values = np.ones(len(row_idx), dtype=np.float32)
    A_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[N, N])
    return tf.sparse.reorder(A_sparse)

A_sparse = create_grid_adjacency_sparse(H, W)  # shape (N, N)

# === 3.  GCN ===
class GraphConvLayer(layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):
        F_in = input_shape[-1]
        self.W = self.add_weight(
            name="W_gcn", 
            shape=(F_in, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name="b_gcn",
            shape=(self.out_features,),
            initializer='zeros',
            trainable=True
        )


        

    def call(self, X):  # X: (batch, N, F)
        def apply_graph_conv(Xb):  # Xb: (N, F)
            AXb = tf.sparse.sparse_dense_matmul(A_sparse, Xb)  # (N, F)
            return tf.matmul(AXb, self.W) + self.b             # (N, out_features)
        return tf.map_fn(apply_graph_conv, X)  # (batch, N, out_features)

# === 4. Model ===
def build_model_convlstm_gnn(input_shape=(12, H, W, 1)):
    inputs = layers.Input(shape=input_shape)

    # 1) Bloc ConvLSTM2D
    x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False)(inputs)
    x = layers.BatchNormalization()(x)  # (batch, H, W, 64)

    # 2) Conversion en graphe
    x = layers.Reshape((N, 64))(x)  # (batch, N, 64)

    # 3) GCN
    x = GraphConvLayer(32)(x)       # (batch, N, 32)
    x = layers.Activation('relu')(x)

    # 4) Retour à l’image
    x = layers.Reshape((H, W, 32))(x)  # (batch, H, W, 32)

    # 5) Sortie : 1 bande
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # (batch, H, W, 1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    model = build_model_convlstm_gnn()
    model.summary()

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# -----------------------------
# Global dimensions
# -----------------------------
H, W = 372, 743
N = H * W

# =====================================================
# 1. Sparse adjacency matrix (8-neighborhood)
# =====================================================
def create_grid_adjacency_sparse_8(h, w):
    row_idx, col_idx = [], []

    # 8-neighborhood 
    neighbors = [
        (-1,  0), (1,  0), (0, -1), (0,  1),   # 4-neighbors
        (-1, -1), (-1, 1), (1, -1), (1,  1)   # diagonals
    ]

    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in neighbors:
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


A_sparse = create_grid_adjacency_sparse_8(H, W)

# =====================================================
# 2. Graph Convolution Layer
# =====================================================
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

    def call(self, X):
        def graph_conv_single(Xb):
            AX = tf.sparse.sparse_dense_matmul(A_sparse, Xb)
            return tf.matmul(AX, self.W) + self.b

        return tf.map_fn(graph_conv_single, X)


# =====================================================
# 3. Transformer Encoder Block
# =====================================================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs)
        attn = self.drop1(attn, training=training)
        x = self.norm1(inputs + attn)

        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)


# =====================================================
# 4. Full ConvLSTM–GCN–Transformer Model
# =====================================================
def build_model_convlstm_gcn_transformer(
    input_shape=(9, H, W, 1)
):
    inputs = layers.Input(shape=input_shape)

    # ---- ConvLSTM ----
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False
    )(inputs)
    x = layers.BatchNormalization()(x)   # (B, H, W, 64)

    # ---- Reshape for GCN ----
    x = layers.Reshape((N, 64))(x)        # (B, N, 64)

    # ---- GCN ----
    x = GraphConvLayer(32)(x)
    x = layers.Activation("relu")(x)

    # ---- Transformer ----
    x = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        ff_dim=64
    )(x)

    # ---- Back to image ----
    x = layers.Reshape((H, W, 32))(x)

    # ---- Output NDVI ----
    outputs = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


# =====================================================
# 5. Test
# =====================================================
if __name__ == "__main__":
    model = build_model_convlstm_gcn_transformer()
    model.summary()

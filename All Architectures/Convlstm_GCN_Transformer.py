import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

H, W = 372, 743
N = H * W

# --- 1.  Sparse Adjacency matrix ---
def create_grid_adjacency_sparse(h, w):
    row_idx, col_idx = [], []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    j = rr * w + cc
                    row_idx.append(i)
                    col_idx.append(j)
    indices = np.stack([row_idx, col_idx], axis=1)
    values = np.ones(len(row_idx), dtype=np.float32)
    A_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[N, N])
    return tf.sparse.reorder(A_sparse)

A_sparse = create_grid_adjacency_sparse(H, W)

# --- 2. layer GCN ---
class GraphConvLayer(layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):
         F_in = input_shape[-1]
         self.W = self.add_weight(name="W_gcn", shape=(F_in, self.out_features), initializer='glorot_uniform', trainable=True)
         self.b = self.add_weight(name="b_gcn", shape=(self.out_features,), initializer='zeros', trainable=True)
       

    def call(self, X):
        def apply_graph_conv(Xb):
            AXb = tf.sparse.sparse_dense_matmul(A_sparse, Xb)
            return tf.matmul(AXb, self.W) + self.b
        return tf.map_fn(apply_graph_conv, X)

# --- 3. Transformer Encoder ---
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):  
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# --- 4. Model ---
def build_model_convlstm_gcn_transformer(input_shape=(12, H, W, 1)):
    inputs = layers.Input(shape=input_shape)

    # ConvLSTM
    x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False)(inputs)
    x = layers.BatchNormalization()(x)  # (batch, H, W, 64)

    # Reshape to (batch, N, 64)
    x = layers.Reshape((N, 64))(x)

    # GCN
    x = GraphConvLayer(32)(x)  # (batch, N, 32)
    x = layers.Activation('relu')(x)

    # Transformer
    x = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x)  # (batch, N, 32)

    # Reshape to image
    x = layers.Reshape((H, W, 32))(x)

    # Sortie
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    model = build_model_convlstm_gcn_transformer()
    model.summary()

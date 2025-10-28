import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# === GCN simplifié en couche personnalisée ===
class GraphConvLayer(layers.Layer):
    def __init__(self, output_dim):
        super(GraphConvLayer, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                 initializer="glorot_uniform",
                                 trainable=True)

        # Création d'une matrice d'adjacence fixe complète (tous les nœuds connectés)
        num_nodes = input_shape[1]
        A = np.ones((num_nodes, num_nodes), dtype=np.float32)
        np.fill_diagonal(A, 1.0)
        A = A / np.maximum(np.sum(A, axis=1, keepdims=True), 1e-6)  # Normalisation
        self.A = tf.constant(A, dtype=tf.float32)

    def call(self, x):
        x = tf.linalg.matmul(self.A, x)
        return tf.matmul(x, self.w)

# === Modèle complet ===
def build_model_gcn_transformer(input_shape=(9, 64, 64, 3)):
    inputs = layers.Input(shape=input_shape)  # (batch, 9, 64, 64, 3)

    # === Bloc CNN par frame
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)  # (batch, 9, 64*64*64)

    # === Réduction temporelle : moyenne
    x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)  # (batch, 64*64*64)

    # === Mise en forme pour le graphe : (4096 nœuds, 64 features)
    x = layers.Reshape((4096, 64))(x)  # (batch, nodes, features)

    # === GCN
    x = GraphConvLayer(32)(x)  # (batch, 4096, 32)

    # === Transformer
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.LayerNormalization()(x)

    # === Reconstruction en image
    x = layers.Reshape((64, 64, 64))(x)
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Test du modèle avec summary ===
if __name__ == "__main__":
    model = build_model_gcn_transformer((9, 64, 64, 3))
    model.summary()

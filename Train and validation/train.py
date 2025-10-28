import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping,ReduceLROnPlateau,Callback
from ConvLstm import build_model_convlstm
from ConvLstm_transformer import build_model_convlstm_transformer
from Convlstm_Gnn import build_model_convlstm_gnn
from Gnn_transformer import build_model_gcn_transformer
from Convlstm_GCN_Transformer import build_model_convlstm_gcn_transformer
import psutil



class RealTimeRAMMonitor(Callback):
    def on_train_batch_end(self, batch, logs=None):
        ram = psutil.virtual_memory()
        loss = logs.get('loss')
        print(f"[Batch {batch+1}] Loss: {loss:.4f} | RAM utilisée : {ram.used / (1024**3):.2f} Go / {ram.total / (1024**3):.2f} Go ({ram.percent}%)")



X_train = np.load("brute_data/X_train.npy")
y_train = np.load("brute_data/y_train.npy")
X_test = np.load("brute_data/X_test.npy")
y_test = np.load("brute_data/y_test.npy")

model = build_model_convlstm(input_shape=X_train.shape[1:])
checkpoint = ModelCheckpoint("brute_data/ConvLstm11.keras", save_best_only=True, monitor="val_loss", mode="min")
csv_logger = CSVLogger("brute_data/training11_log.csv", append=True)
ReduceLR=ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
Earlystop= EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=1, batch_size=1,
                    callbacks=[checkpoint, csv_logger,Earlystop,ReduceLR])

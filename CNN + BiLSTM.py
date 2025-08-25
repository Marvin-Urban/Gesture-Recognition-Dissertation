import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn_bilstm(input_steps: int = 16, n_channels: int = 7, n_classes: int = 18) -> tf.keras.Model:
    inp = layers.Input(shape=(input_steps, n_channels))
    x = layers.Conv1D(32, 5, padding="same", activation="relu", kernel_initializer="he_uniform")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=False))(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inp, out, name="cnn_bilstm")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

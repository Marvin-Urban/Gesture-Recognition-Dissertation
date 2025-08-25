import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_fcnn(input_dim: int, num_classes: int = 18) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inp, out, name="fcnn_baseline")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers,models
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(input_shape: tuple) -> Model:

    ### Model Instantiation
    model = models.Sequential()

    ####################################
    #       0 - Preprocessing          #
    ####################################

    model.add(layers.Resizing(224, 224, input_shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)))

    ####################################
    #        1 - Convolutions          #
    ####################################

    ### 1st Convolution & MaxPooling
    model.add(layers.Conv2D(32, kernel_size=(5,5), activation="relu", input_shape=(4487,416,416,3)[1:]))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ####################################
    # 3 - Flatten, Dense & Last layers #
    ####################################

    ### Flatten layer
    model.add(layers.Flatten())

    ### Dense layer(s)
    model.add(layers.Dense(30, activation='relu'))

    ### Last layer
    model.add(layers.Dense(26, activation='softmax'))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Model compilation
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    # $CODE_BEGIN
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    # $CODE_END

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

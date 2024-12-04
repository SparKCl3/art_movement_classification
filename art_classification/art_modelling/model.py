import tensorflow as tf
import os
import time
import glob
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model as keras_load_model

local_registry_path = os.environ.get("LOCAL_REGISTRY_PATH")
local_registry_path_checkpoint = os.environ.get("LOCAL_REGISTRY_PATH_CHEKPOINT")

# Function to load dataset from a directory
def get_from_directory(folder_path_type, batch_size, color_mode, image_size, crop_to_aspect_ratio, seed=0, subset=None, validation_split=None):
    dataset = image_dataset_from_directory(
        folder_path_type,
        batch_size=batch_size,
        color_mode=color_mode,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        seed=seed,
        subset=subset,
        validation_split=validation_split,
        crop_to_aspect_ratio=crop_to_aspect_ratio
    )
    return dataset

# Baseline CNN model
########################################################################
def baseline_cnn_model(input_shape: tuple) -> Model:
    # Model Instantiation
    model = models.Sequential()

    ####################################
    #       0 - Preprocessing          #
    ####################################
    model.add(layers.Resizing(224, 224, input_shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    ####################################
    #        1 - Convolutions          #
    ####################################
    # 1st Convolution & MaxPooling
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ####################################
    # 3 - Flatten, Dense & Last layers #
    ####################################
    # Flatten layer
    model.add(layers.Flatten())

    # Dense layer(s)
    model.add(layers.Dense(30, activation='relu'))

    # Last layer
    model.add(layers.Dense(os.environ.get('NUM_CLASSES'), activation='softmax'))

    print("✅ Model initialized")
    return model

# Funnel (\/) CNN model
########################################################################
def cnn_model_funnel(input_shape: tuple) -> Model:
    # Model Instantiation
    model = models.Sequential()

    ####################################
    #       0 - Preprocessing          #
    ####################################
    model.add(layers.Resizing(224, 224, input_shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    ####################################
    #        1 - Convolutions          #
    ####################################
    # 1st Convolution & MaxPooling
    model.add(layers.Conv2D(128, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 2nd Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 3rd Convolution & MaxPooling
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ####################################
    # 3 - Flatten, Dense & Last layers #
    ####################################
    # Flatten layer
    model.add(layers.Flatten())

    # Dense layer(s)
    model.add(layers.Dense(30, activation='relu'))
    print(f"Nombre de classe NUM_CLASSES {os.environ.get('NUM_CLASSES')}")


    # Last layer
    model.add(layers.Dense(os.environ.get('NUM_CLASSES'), activation='softmax'))

    print("✅ Model initialized")
    return model

# Inverted funnel (/\) CNN model
########################################################################
def cnn_model_inverted_funnel(input_shape: tuple) -> Model:
    # Model Instantiation
    model = models.Sequential()

    ####################################
    #       0 - Preprocessing          #
    ####################################
    model.add(layers.Resizing(224, 224, input_shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    ####################################
    #        1 - Convolutions          #
    ####################################

    # 1st Convolution & MaxPooling
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 2nd Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 3rd Convolution & MaxPooling
    model.add(layers.Conv2D(128, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ####################################
    # 3 - Flatten, Dense & Last layers #
    ####################################
    # Flatten layer
    model.add(layers.Flatten())

    # Dense layer(s)
    model.add(layers.Dense(30, activation='relu'))

    # Last layer
    model.add(layers.Dense(os.environ.get('NUM_CLASSES'), activation='softmax'))

    print("✅ Model initialized")
    return model

# H CNN model
########################################################################
def cnn_model_h(input_shape: tuple) -> Model:
    # Model Instantiation
    model = models.Sequential()

    ####################################
    #       0 - Preprocessing          #
    ####################################
    model.add(layers.Resizing(224, 224, input_shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    ####################################
    #        1 - Convolutions          #
    ####################################

    # 1st Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 2nd Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 3rd Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ####################################
    # 3 - Flatten, Dense & Last layers #
    ####################################
    # Flatten layer
    model.add(layers.Flatten())

    # Dense layer(s)
    model.add(layers.Dense(30, activation='relu'))

    # Last layer
    model.add(layers.Dense(os.environ.get('NUM_CLASSES'), activation='softmax'))

    print("✅ Model initialized")
    return model

# MODELE RESNET
########################################################################
def resnet_block(X,filters, strides):

    x_skip=X
#conv1
    X=layers.Conv2D(filters=filters, kernel_size=3,strides=strides, padding='same')(X)
    X=layers.BatchNormalization()(X)
    X=layers.ReLU()(X)
#conv 2
    X=layers.Conv2D(filters=filters, kernel_size=3,strides=1, padding='same')(X)
    X=layers.BatchNormalization()(X)

    # permet d'ajuster la taille de la fuite entre deux changement de taille de filtres
    if strides != 1 or X.shape[-1] != x_skip.shape[-1]:
        x_skip = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same")(x_skip)
        x_skip = layers.BatchNormalization()(x_skip)

#Additionne X et le residu
    X=layers.Add()([X,x_skip])
    X=layers.ReLU()(X)
    print(type(X))
    return X

def initialize_resnet_model(classes,input_shape):

    X_input=layers.Input(input_shape)

    #Conv Init
    X=layers.Conv2D(64,kernel_size=7,strides=2,padding='same')(X_input)
    X=layers.BatchNormalization()(X)
    X=layers.ReLU()(X)
    X=layers.MaxPool2D(pool_size=3, strides=2)(X)

    #Res1 2 blocs, 64 filtres
    X=resnet_block(X,filters=64, strides=1)
    X=resnet_block(X,filters=64, strides=1)
    #Res1 2 blocs, 128 filtres
    X=resnet_block(X,filters=128, strides=2)
    X=resnet_block(X,filters=128, strides=1)
    #Res1 2 blocs, 256  filtres
    X=resnet_block(X,filters=256, strides=2)
    X=resnet_block(X,filters=256, strides=1)
    #Res1 2 blocs, 512 filtres
    X=resnet_block(X,filters=512, strides=2)
    X=resnet_block(X,filters=512, strides=1)
    X=layers.GlobalAveragePooling2D()(X)
    X=layers.Dense(classes,activation='softmax')(X)
    model=models.Model(inputs=X_input,outputs=X,name='ResnetInput')
    return model


########################################################################

# Function to compile the model
def compile_model(model: Model, learning_rate: float) -> Model:
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

def train_model(model: Model, train_ds, val_ds, epochs, patience):
    """
    Train the model with early stopping and checkpointing, using a validation dataset.

    Args:
        model (Model): The Keras model to train.
        train_ds: The training dataset.
        val_ds: The validation dataset.
        epochs (int): Number of epochs to train.
        patience (int): Number of epochs with no improvement to wait before stopping.

    Returns:
        model (Model): The trained Keras model.
        history (History): The training history.
    """
    # Early stopping to monitor validation loss
    es = EarlyStopping(
        monitor='val_loss',  # Monitors validation loss for early stopping
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Checkpoint callback for saving model checkpoints
    checkpoint_callback = create_checkpoint()

    # Fit the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,  # Use the separate validation dataset
        epochs=epochs,
        callbacks=[es, checkpoint_callback],
        verbose=1
    )

    return model, history


# Function to evaluate the model
def evaluate_model(model: Model, test_ds):
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        test_ds,
        verbose=0,
        return_dict=True
    )
    print(metrics)
    return metrics


def create_checkpoint(checkpoint_path=local_registry_path_checkpoint):

    """
    Creates a checkpoint directory if it doesn't exist and returns a ModelCheckpoint callback.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create the directory if it does not exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the ModelCheckpoint callback
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )
    print(f"✅ Checkpoint callback created at {checkpoint_path}")
    return cp_callback

def load_model(local_registry_path=local_registry_path) -> "Model":
    """
    Loads the best model (based on `val_accuracy`) from the specified directory.
    """
    if not os.path.exists(local_registry_path):
        print(f"⚠️ No models directory found at {local_registry_path}")
        return None

    # Get all saved model files in the directory
    model_files = glob.glob(f"{local_registry_path}/*.h5")
    if not model_files:
        print(f"⚠️ No model files found in {local_registry_path}")
        return None

    # Select the most recent model file
    most_recent_model_path = sorted(model_files)[-1]
    latest_model = keras_load_model(most_recent_model_path)

    print(f"✅ Model loaded from {most_recent_model_path}")
    return latest_model


def save_model(model, local_registry_path=local_registry_path, val_accuracy=None, test_ds=None):
    """
    Saves the model to a specified directory if it outperforms the existing model.
    If the directory does not exist, it will be created.
    """
    import numpy as np

    # Ensure the directory exists
    os.makedirs(local_registry_path, exist_ok=True)

    # Create a timestamped filename for the model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(local_registry_path, f"model_{timestamp}.h5")

    # Get the list of saved models
    model_files = glob.glob(f"{local_registry_path}/*.h5")

    if model_files:
        # Load the most recent model
        saved_most_recent_model_path = sorted(model_files)[-1]
        saved_latest_model = keras_load_model(saved_most_recent_model_path)

        # Evaluate the existing model on the test dataset
        existing_metrics = evaluate_model(saved_latest_model, test_ds)
        existing_accuracy = existing_metrics.get("accuracy", -np.inf)  # Default to -inf if unavailable

        # Evaluate the current model
        new_metrics = evaluate_model(model, test_ds)
        new_accuracy = new_metrics.get("accuracy", -np.inf)

        # Compare accuracies
        if new_accuracy > existing_accuracy:
            # Save the new model if it's better
            model.save(model_path)
            print(f"✅ New model saved locally at {model_path} with accuracy: {new_accuracy:.4f}")
        else:
            print(f"❌ Model not saved. Existing model has better or equal accuracy ({existing_accuracy:.4f}).")
    else:
        # Save the model if no previous model exists
        model.save(model_path)
        print(f"✅ Model saved locally at {model_path} (no existing model to compare).")

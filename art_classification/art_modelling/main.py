from preprocessing import import_data_from_bucket, process_and_resize_image
from model import (
    get_from_directory,
    baseline_cnn_model,
    cnn_model_funnel,
    cnn_model_inverted_funnel,
    cnn_model_h,
    initialize_resnet_model,
    compile_model,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
import os

# Paths
# split_dataset_path = 'split_dataset'
# train_path = os.path.join(split_dataset_path, 'train')
# val_path = os.path.join(split_dataset_path, 'val')
# test_path = os.path.join(split_dataset_path, 'test')

# Check if local dataset exists
# if not os.path.exists(split_dataset_path):
#     print(f"Local dataset not found at {split_dataset_path}. Importing from bucket...")
#     import_data_from_bucket()
# else:
#     print(f"Local dataset found at {split_dataset_path}.")

# Environment Variables
batch_size = int(os.environ.get('BATCH_SIZE', 32))
num_classes = int(os.environ.get('NUM_CLASSES', 26))  # Defaulting to 26 classes
epochs = int(os.environ.get('EPOCHS', 10))  # Defaulting to 10 epochs
patience = int(os.environ.get('PATIENCE', 3))  # Defaulting to 3 epochs patience
learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))  # Defaulting to 0.001 learning rate
crop_to_aspect_ratio = bool(os.environ.get("CROP_TO_ASPECT_RATIO", False))
input_shape = tuple(map(int, os.environ.get("INPUT_SHAPE", "416,416,3").split(',')))
'''
# Preprocessing function
def preproc_tts():
    # TensorFlow Dataset Preparation
    train_ds = get_from_directory(
        train_path,
        batch_size=batch_size,
        color_mode='rgb',
        image_size=(416, 416),
        validation_split=0.3,
        seed=0,
        subset='training',
        crop_to_aspect_ratio=crop_to_aspect_ratio
    )

    val_ds = get_from_directory(
        val_path,
        batch_size=batch_size,
        color_mode='rgb',
        image_size=(416, 416),
        validation_split=0.3,
        seed=0,
        subset='validation',
        crop_to_aspect_ratio=crop_to_aspect_ratio
    )

    test_ds = get_from_directory(
        test_path,
        batch_size=batch_size,
        color_mode='rgb',
        image_size=(416, 416),
        crop_to_aspect_ratio=crop_to_aspect_ratio
    )

    return train_ds, val_ds, test_ds

'''

def preproc_tts():
# Import data
    imported_data = import_data_from_bucket()

# Train / Test
    train_ds, test_ds = get_from_directory(
        imported_data,
        batch_size=batch_size,
        color_mode='rgb',
        image_size=(416, 416),
        validation_split=0.2,
        seed=0,
        subset='both',
        crop_to_aspect_ratio=crop_to_aspect_ratio)

# Train / Val
    validation_size = int(0.2 * len(train_ds))
    train_ds = train_ds.skip(validation_size)
    val_ds = train_ds.take(validation_size)

    return train_ds, test_ds, val_ds

# Run preprocessing
# print("Datasets are ready!")

# OLD ------ v

# train_ds = get_from_directory(folder_path_train, batch_size, 'rgb', image_size=(416, 416))
# val_ds = get_from_directory(folder_path_val, batch_size, 'rgb', image_size=(416, 416))
# assert len(val_ds.class_names) == num_classes, "Number of classes in validation dataset mismatch"
# test_ds = get_from_directory(folder_path_test, batch_size, 'rgb', image_size=(416, 416))

# OLD ------ ^
input_shape = (416, 416, 3)  # Include channels for RGB

def train(input_shape, learning_rate, train_ds, val_ds):
    '''
    1. Model Initialization
    2. Compilation
    3. Training
    4. Save model
    5. Return validation accuracy
    '''

    # --- 1 --- Model Initialization ---------

    model_type = os.environ.get('MODEL_TYPE')

    if model_type == 'resnet':
        model = initialize_resnet_model(classes=num_classes, input_shape=input_shape)
    elif model_type == 'cnn-model-funnel':
        model = cnn_model_funnel(input_shape=input_shape)
    elif model_type == 'cnn-model-inverted-funnel':
        model = cnn_model_inverted_funnel(input_shape=input_shape)
    elif model_type == 'cnn-model-h':
        model = cnn_model_h(input_shape=input_shape)
    else:
        model = baseline_cnn_model(input_shape=input_shape)

    # --- 2 --- Compilation ---------

    model = compile_model(model, learning_rate)

    # --- 3 --- Training ---------

    model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        patience=patience
    )

    print("Training History:", history.history)

    # --- 4 --- Accuracy ---------

    history_data = history.history

    # Mean accuracy for training data
    mean_train_accuracy = sum(history_data['accuracy']) / len(history_data['accuracy'])

    # Mean accuracy for validation data (if available)
    mean_val_accuracy = None
    if 'val_accuracy' in history_data:
        mean_val_accuracy = sum(history_data['val_accuracy']) / len(history_data['val_accuracy'])

    # --- 5 --- Save model ---------

    return mean_train_accuracy, mean_val_accuracy, model

def summary_evaluate(model, test_ds):
    '''
    Summarizes the model and evaluates it on the test dataset.
    '''
    model.summary()
    print("Evaluating model on test dataset...")
    metrics_dict = evaluate_model(model, test_ds)
    print("Evaluation Metrics:", metrics_dict)
    return metrics_dict


 #saved_model = save_model(model, local_registry_path="models/model", test_ds=test_ds)

# Model Evaluation
#def load_evaluate(saved_model,test_ds):
   #load model
    #if os.environ.get('MODEL_ACTION')==True:
    #loaded_model = load_model(local_registry_path="models/model")
    #print("Loaded Model Summary:", loaded_model.summary())
    #evaluate model
    #metrics_dict = evaluate_model(saved_model, test_ds)
    # Display Results
   # print("Evaluation Metrics:", metrics_dict)

#main
if __name__ == '__main__':
    import_data_from_bucket()
    # Preprocess the dataset
    train_ds, val_ds, test_ds = preproc_tts()

    # Train the model
    mean_train_accuracy, mean_val_accuracy, model = train(input_shape, learning_rate, train_ds, val_ds)

    # Evaluate the model
    metrics_dict = evaluate_model(model, test_ds)
    print(metrics_dict)

    # Output results
    #print(f"Mean Training Accuracy: {mean_train_accuracy}")
    #print(f"Mean Validation Accuracy: {mean_val_accuracy}")
    #print(f"Evaluation Metrics: {metrics_dict}")


    # transforming predicted image
    input_image_path = "/Users/clothildemorin/code/SparKCl3/art_movement_classification/test_image/exemple_image.jpg"
    resized_image_array = process_and_resize_image(input_image_path)

    # Prédiction avec le modèle
    predictions = model.predict(resized_image_array)
    print(f"Model predictions: {predictions}")

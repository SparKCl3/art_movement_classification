from preprocessing import import_data, import_data_from_bucket
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

# OLD ------ v

# Preprocessing - Load dataset paths
# folder_path_train = import_data('train')
# folder_path_test = import_data('test')
# folder_path_val = import_data('valid')

# OLD ------ ^

#folder_path_train = os.environ.get('LOCAL_REGISTRY_PATH')+'train'
#folder_path_test = os.environ.get('LOCAL_REGISTRY_PATH')+'test'
#folder_path_val = os.environ.get('LOCAL_REGISTRY_PATH')+'valid'


# Environment Variables
batch_size = int(os.environ.get('BATCH_SIZE', 32))
num_classes = int(os.environ.get('NUM_CLASSES', 26))  # Defaulting to 26 classes
epochs = int(os.environ.get('EPOCHS', 10))  # Defaulting to 10 epochs
patience = int(os.environ.get('PATIENCE', 3))  # Defaulting to 3 epochs patience
learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))  # Defaulting to 0.001 learning rate
crop_to_aspect_ratio = os.environ.get("CROP_TO_ASPECT_RATIO")
input_shape = os.environ.get("INPUT_SHAPE")


#Definition Preprocessing :

def preproc_tts():
# Import data
    imported_data = import_data_from_bucket()

# TensorFlow Dataset Preparation
    train_ds, test_ds = get_from_directory(
        imported_data,
        batch_size=batch_size,
        color_mode='rgb',
        image_size=(416, 416),
        validation_split=0.3,
        seed=0,
        subset='both',
        crop_to_aspect_ratio=crop_to_aspect_ratio)

    assert len(train_ds.class_names) == num_classes, "Number of classes in train dataset mismatch"
    assert len(test_ds.class_names) == num_classes, "Number of classes in test dataset mismatch"
    print("✅ preprocess_tts() done")

    return train_ds, test_ds

# OLD ------ v

# train_ds = get_from_directory(folder_path_train, batch_size, 'rgb', image_size=(416, 416))
# val_ds = get_from_directory(folder_path_val, batch_size, 'rgb', image_size=(416, 416))
# assert len(val_ds.class_names) == num_classes, "Number of classes in validation dataset mismatch"
# test_ds = get_from_directory(folder_path_test, batch_size, 'rgb', image_size=(416, 416))

# OLD ------ ^

def train(input_shape, learning_rate, train_ds, test_ds, validation_split=0.3):
    '''
    1. Model Initialization
    2. Compilation
    3. Training
    4. Save model
    5. Return val accuracy
    '''

    # --- 1 --- Model Initialization ---------

    model_type = os.environ.get('MODEL_TYPE')

    if model_type=='resnet':
        model = initialize_resnet_model(classes=num_classes,input_shape=input_shape)
    elif model_type=='cnn-model-funnel':
        model = cnn_model_funnel(input_shape=input_shape)
    elif model_type=='cnn-model-inverted-funnel':
        model = cnn_model_inverted_funnel(input_shape=input_shape)
    elif model_type=='cnn-model-h':
        model = cnn_model_h(input_shape=input_shape)
    else:
        model = baseline_cnn_model(input_shape=input_shape)

    # --- 2 --- Compilation ---------

    model = compile_model(model, learning_rate)

    # --- 3 --- Training ---------

    model, history = train_model(
    model=model,
    train_ds=train_ds,
    epochs=epochs,
    validation_split=validation_split,
    patience=patience
    )

    print("Training History:", history.history)

    # --- 4 --- Accuracy ---------

    history_data = history.history

    # Mean accuracy for training data
    mean_train_accuracy = sum(history_data['accuracy']) / len(history_data['accuracy'])

    # Mean accuracy for validation data (if available)
    if 'val_accuracy' in history_data:
        mean_val_accuracy = sum(history_data['val_accuracy']) / len(history_data['val_accuracy'])

    # --- 5 --- Save model ---------

    return mean_train_accuracy, mean_val_accuracy, model

def summary_evaluate(model, test_ds):
    model_summary =  model.summary()
    print("Model Summary:", model_summary)
    metrics_dict = evaluate_model(model, test_ds)
    print("Evaluation Metrics:", metrics_dict)
    return model_summary, metrics_dict


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
    train_ds, test_ds = preproc_tts()
    mean_train_accuracy, mean_val_accuracy, model = train(input_shape, learning_rate, train_ds, test_ds, validation_split=0.3)
    model_summary, metrics_dict = summary_evaluate(model, test_ds)
    print(model_summary, metrics_dict)

    # transforming predicted image
    # input_image_path = "/Users/clothildemorin/code/SparKCl3/art_movement_classification/test_image/exemple_image.jpg"
    # output_image_path = "/Users/clothildemorin/code/SparKCl3/art_movement_classification/test_image/resize_image.jpg"
    # image_array, resized_image = process_and_resize_image(input_image_path, output_image_path)

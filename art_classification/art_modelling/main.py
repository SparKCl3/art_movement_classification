from preprocessing import import_data
from model import (
    get_from_directory,
    initialize_model,
    compile_model,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
import os

# Preprocessing - Load dataset paths
folder_path_train = import_data('train')
folder_path_test = import_data('test')
folder_path_val = import_data('valid')

# Environment Variables
batch_size = int(os.environ.get('BATCH_SIZE', 32))
num_classes = int(os.environ.get('NUM_CLASSES', 26))  # Defaulting to 26 classes
epochs = int(os.environ.get('EPOCHS', 10))  # Defaulting to 10 epochs
patience = int(os.environ.get('PATIENCE', 3))  # Defaulting to 3 epochs patience
learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))  # Defaulting to 0.001 learning rate

# TensorFlow Dataset Preparation
train_ds = get_from_directory(folder_path_train, batch_size, 'rgb', image_size=(416, 416))
assert len(train_ds.class_names) == num_classes, "Number of classes in train dataset mismatch"
val_ds = get_from_directory(folder_path_val, batch_size, 'rgb', image_size=(416, 416))
assert len(val_ds.class_names) == num_classes, "Number of classes in validation dataset mismatch"
test_ds = get_from_directory(folder_path_test, batch_size, 'rgb', image_size=(416, 416))
assert len(test_ds.class_names) == num_classes, "Number of classes in test dataset mismatch"

# Model Initialization
input_shape = (416, 416, 3)
model = initialize_model(input_shape=input_shape)

# Model Compilation
model = compile_model(model, learning_rate)

# Model Training
model, history = train_model(
    model=model,
    train_ds=train_ds,
    epochs=epochs,
    validation_data=val_ds,
    patience=patience
)

# Model Evaluation
metrics_dict = evaluate_model(model, test_ds)

# Display Results
print("Training History:", history.history)
print("Evaluation Metrics:", metrics_dict)

# Save the Model
save_model(model, local_registry_path="models/model", test_ds=test_ds)

if os.environ.get('MODEL_ACTION')==True:
    loaded_model = load_model(local_registry_path="models/model")
    print("Loaded Model Summary:", loaded_model.summary())

#main
if __name__ == '__main__':
    pass

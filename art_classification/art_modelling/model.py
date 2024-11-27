from tensorflow.keras.utils import image_dataset_from_directory

def get_from_directory(folder_path_type, batch_size, color_mode, image_size):
    dataset = image_dataset_from_directory(
        folder_path_type,
        color_mode=color_mode,
        labels='inferred',
        image_size=image_size,
        label_mode='categorical',
        batch_size=batch_size,
    )
    return dataset

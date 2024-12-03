import numpy as np
import pandas as pd
from pathlib import Path

from google.cloud import storage
import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

def import_data_from_bucket():

    bucket_name = os.environ.get('BUCKET_NAME')
    destination_folder = os.environ.get('DESTINATION_FOLDER')

    # Initialiser le client Google Cloud Storage
    client = storage.Client()

    # Accéder au bucket
    bucket = client.bucket(bucket_name)

    print(f'Bucket: {bucket}')

    # Lister les objets dans le bucket
    blobs = bucket.list_blobs()

    # print(f'Blob list: {list(blobs)}')

    # Define the directory path in the home directory
    dir_path = os.path.expanduser(destination_folder)
    # Create a folder if no existing one
    if not os.path.isdir(dir_path):
        # Create the directory
        os.mkdir(dir_path)

    print(f"Downloading dataset from 'gs://{bucket_name}' to '{destination_folder}'...")

    for blob in blobs:
        # Create local path in VM
        local_path = os.path.join(dir_path, blob.name)

        if not os.path.isdir(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Download file
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} to {local_path}")
            else:
                pass

    return f"{dir_path}/Dataset_aug/"

###############################################################

'''
# Paths
base_path = os.path.expanduser("~/code/SparKCl3/art_movement_classification/dataset")
output_dir = os.path.expanduser("~/code/SparKCl3/art_movement_classification/split_dataset")

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create directories for the splits
splits = ['train', 'val', 'test']
for split in splits:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# Split the dataset
for class_name in os.listdir(base_path):
    class_dir = os.path.join(base_path, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip non-folder files

    # Get all file paths for the current class
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

    # Train-Test-Val split
    train_files, test_files = train_test_split(files, test_size=(val_ratio + test_ratio), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Define the destination paths
    for file_path, split in zip([train_files, val_files, test_files], splits):
        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        # Move the files
        for file in file_path:
            shutil.copy(file, split_class_dir)

print("Dataset splitting complete!")'''
###############################################################

def process_and_resize_image(input_image_path, output_image_path=None, target_size=(416, 416)):
    image = Image.open(input_image_path)
    image = image.convert("RGB")

    print(f"Original image size: {image.size}")  # Debugging

    # Redimensionner l'image
    resized_image = image.resize(target_size)
    print(f"Resized image size: {resized_image.size}")  # Debugging

    resized_image = image.resize(target_size)
    print(f"Resized image size: {resized_image.size}")  # Debugging
    if output_image_path:
        resized_image.save(output_image_path)
        print(f"Resized image saved to: {output_image_path}")

    image_array = np.array(resized_image) / 255.0

    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

#################################################################

def path_df(folder_path): #
    chemin_dataset = Path(folder_path)
    data = []
    for dossier in chemin_dataset.iterdir():
        if dossier.is_dir():
            classe = dossier.name
            for fichier in dossier.iterdir():
                if fichier.is_file():
                    chemin_fichier = fichier.as_posix()
                    data.append({"path": chemin_fichier, "class": classe})

    df = pd.DataFrame(data)

    return df

# OLD -------

# def import_data(dataset):
#     return f"././notebook/wiki-art-1/{dataset}"

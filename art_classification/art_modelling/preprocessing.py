import numpy as np
import pandas as pd
from pathlib import Path

from google.cloud import storage
import os

def import_data_from_bucket():

    bucket_name = os.environ.get('BUCKET_NAME')
    destination_folder = os.environ.get('DESTINATION_FOLDER')

    # Initialiser le client Google Cloud Storage
    client = storage.Client()

    # Accéder au bucket
    bucket = client.bucket(bucket_name)

    # print(f'Bucket: {bucket}')

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

    return f"{dir_path}Dataset_aug/"


###############################################################

def process_and_resize_image(input_image_path, output_image_path=None, target_size=(416, 416)):
#working function without the streamlit object - need to be modified later
    image = Image.open(input_image_path)
    print(f"Original image size: {image.size}") #to comment during prod
    resized_image = image.resize(target_size)
    print(f"Resized image size: {resized_image.size}") #to comment during prod
    image_array = np.array(resized_image)
    if output_image_path:
        resized_image.save(output_image_path)
        print(f"Resized image saved to: {output_image_path}")
    return image_array, resized_image

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

def import_data(dataset):
    return f"././notebook/wiki-art-1/{dataset}"

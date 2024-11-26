import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import cv2


# Import de la donnée <= le lien sera à modifier pour aller chercher la donnée
# dans le bucket approprié
def import_data():
    return "././notebook/wiki-art-1/train" # lien relatif : le dossier contenant le train set est dans le répertoire du projet



# Création d'un dataframe avec le chemin du fichier de donnée et la classe correspondante
# Prend en entrée le path du répertoire de donnée (qu'on obtiendra avec la fonction import_data())
def path_df(folder_path): #
    chemin_dataset = Path(folder_path)
    data = []
    for dossier in chemin_dataset.iterdir():
        if dossier.is_dir():  # Vérifier que c'est un dossier
            classe = dossier.name  # Le nom du dossier est la classe
            # Parcourir les fichiers dans ce dossier
            for fichier in dossier.iterdir():
                if fichier.is_file():  # Vérifier que c'est un fichier
                    chemin_fichier = fichier.as_posix()  # Chemin absolu du fichier
                    data.append({"path": chemin_fichier, "class": classe})

    df = pd.DataFrame(data)

    return df



# Création d'un array contenant toutres les images (shape des images du petit dataset : (416, 416, 3))
# Prend en argument le Dataframe obtenu avec la fonction path_df()
def img_array(df):
    return np.array([cv2.imread(path) for path in df.path])


# Fonction de preprocessing selon la méthode choisie en argument (1, 2 ou  3)

def preprocessing(method:int):
    '''
    Choisir la méthode de preprocessing :
    1. valeurs / 255
    2. Normalisation des valeurs
    3. Sans preprocessing mais en utilisant un tenseur
    '''
    URL = import_data()
    df = path_df(URL)
    img_arr = img_array(df)

    std = img_arr.std()
    mean= img_arr.mean()

    if method == 1:
        preproc = img_arr/255

    elif method == 2:
        preproc = (img_arr - mean)/std

    elif method == 3:
        preproc = image_dataset_from_directory(
            URL,
            labels="inferred",
            label_mode="categorical",
            image_size=(416, 416), # resize on the fly
            batch_size=32
            )
    else :
        return "Wrong preprocessing method"

    return preproc

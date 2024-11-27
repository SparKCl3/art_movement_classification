import numpy as np
import pandas as pd
from pathlib import Path
#import cv2

def import_data(dataset):
    return f"././notebook/wiki-art-1/{dataset}"


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

def preprocessing(type):
    folder_path = import_data(type)
    df = path_df(folder_path)
    return df

#def img_array(df):
    #return np.array([cv2.imread(path) for path in df.path])


# Fonction de preprocessing selon la méthode choisie en argument (1, 2 ou  3)
# et le dossier de donnée ("train", "test", "valid")

#def preprocessing(dataset:str,method:int):
    '''
    Choisir la méthode de preprocessing :
    1. valeurs / 255
    2. Normalisation des valeurs
    3. Sans preprocessing mais en utilisant un tenseur
    '''
    #URL = import_data(dataset)
    #df = path_df(URL)
    #img_arr = img_array(df)

    #std = img_arr.std()
    #mean= img_arr.mean()

    #if method == 1:
        #preproc = img_arr/255

    #elif method == 2:
        #preproc = (img_arr - mean)/std

    #else :
        #return "Wrong preprocessing method"

    #return preproc

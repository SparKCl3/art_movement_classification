import numpy as np
import pandas as pd
from pathlib import Path

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

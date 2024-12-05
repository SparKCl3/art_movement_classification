from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from art_classification.art_modelling.model import load_model
from art_classification.art_modelling.preprocessing import process_and_resize_image
import os
import numpy as np
from PIL import Image
from tensorflow import convert_to_tensor


# --- 0 --- Env variable(s)
output_image_path=os.environ.get('OUTPUT_IMAGE_PATH')

# --- 0 --- Create the FastAPI app
app = FastAPI()


@app.post("/upload/")
def get_image(file: UploadFile = File(...)):

    # Get, preprocess & and save image
    preprocessed_image = process_and_resize_image(file.file,output_image_path=output_image_path)
    print(type(preprocessed_image))


@app.get("/predict/")
def predict():
    # --- 1 --- Load the model
    model = load_model()

    # --- 2 --- Labels dictionary
    labels = {"1": "Abstract Expressionism", "2": "Impressionism", "3": "Pointillism Fauvism", "4": "Renaissance", "5": "Ukiyo e", "6": "Art Nouveau Modern", "7": "Minimalism", "8": "Pop Art", "9": "Rococo", "10": "Baroque", "11": "Naive Art Primitivism", "12": "Post Impressionism", "13": "Romanticism", "14": "Cubism", "15": "Eo Impressionism", "16": "Pre Raphaelite Brotherhood", "17": "Surrealism", "18": "Expressionism", "19": "Neoclassicism", "20": "Realism", "21": "Symbolism"}

    # --- 3 --- Get image and convert to tensor
    im = Image.open(output_image_path + '/image.jpeg')
    im_array = np.array(im)
    im_array = np.array(im_array) / 255.0
    im_array = np.expand_dims(im_array, axis=0)
    im_tensor = convert_to_tensor(im_array)

    # --- 4 --- Prediction
    art_movement_prediction = model.predict(im_tensor)
    class_pred = str(np.argmax(art_movement_prediction))
    final_prediction = labels[class_pred]

    print(f"Model prediction: {final_prediction}")

    return final_prediction

#######################

@app.get("/")
def read_root():
    return {"message": "Connected API"}

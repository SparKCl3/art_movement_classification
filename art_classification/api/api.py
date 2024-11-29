from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Connected API"}


#functions pour le main
#processing image api
#model
#finetuner model
#docker
#api
#streamlit
#slides

#pip install fastapi uvicorn

#import following
from statistics import variance
import uvicorn #ASGI
from fastapi import FastAPI
from banknotes import BankNote
import pickle
import pandas as pd

#Create the app project
app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message' : 'Hello, World'}

#Route with a single parameter, returns the parameter with a message
# Located ar http://127.0.0.1:8000/Anyname

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to API World' : f'{name}'}

#run the api with uvicorn
# will run on http://127.0.0.1:8000

@app.post('/predict')
#data will capture value
def predict_banknote(data:BankNote):
    
    data = data.dict()
    print(data)
    variance = data['variance']
    print(variance)
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy  = data['entropy']
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)

    if(prediction[0]>0.5):
        prediction = "Fake Note"
    else:
        prediction = "Bank Note"

    return {
        'prediction' : prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn main:app --reload
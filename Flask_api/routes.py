from flask import Flask, request

from flask_cors import CORS

from main import predict

app = Flask(__name__)
CORS(app)

@app.route("/hi", methods=["GET"])
def hi():
    return {"response": "Hey there!"}

@app.route("/predict", methods=["POST"])
def handlePrediction():

    body = request.get_json()

    if("birads" not in body):
        return generateResponse(400, "É obrigatório selecionar uma avaliação BI-RADS!")

    if("dateOfBirth" not in body):
        return generateResponse(400, "É obrigatório selecionar uma data de nascimento!")

    if("forma" not in body):
        return generateResponse(400, "É obrigatório selecionar uma forma!")

    if("margem" not in body):
        return generateResponse(400, "É obrigatório selecionar uma margem!")

    if("densidade" not in body):
        return generateResponse(400, "É obrigatório selecionar uma densidade!")

    prediction = predict(body["birads"], body["dateOfBirth"], body["forma"], body["margem"], body["densidade"])

    return generateResponse(200, "Resultado obtido com sucesso!", "prediction", prediction) 

def generateResponse(status, message, resultName=False, result=False):
    response ={}
    response["status"] = status
    response["message"] = message

    if(resultName and result):
        response[resultName] = result
    
    return response

app.run()
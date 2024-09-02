from flask import Flask, render_template, jsonify, request
#from flask_cors import CORS, cross_origin
from src.rag import * # Import all defined functions
import joblib
import numpy as np
import pandas as pd
from src.model import NeuralNet
from src.Sentence_Processing import *
from src import *
import json 
import torch
import random
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain.vectorstores import Pinecone
import pinecone
import os 
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
import tensorflow as tf 
from data.desases import diseases


app = Flask(__name__)

#English

#Home page
@app.route("/")
@app.route("/Home.html")
def home():
    return render_template('Home.html')


#Fertify
@app.route("/Fertify.html", methods=["GET", "POST"])
def Fertify():
    return render_template('Fertify.html')

@app.route("/predict",methods=["POST","GET"])
def predict_fertilizer():
    # loading the model 
    model = joblib.load('./models/Fertify.pkl')
    # Get the data from the POST request
    input_data = request.get_json(force=True)

    input_data = input_data['input_data']
    
    # Convert data into numpy array
    input_data_df = pd.DataFrame(input_data)
    print(input_data_df)
    
     #Make prediction
    prediction = model.predict(input_data_df)
    print(f"prediction {prediction}")
    prediction = prediction[0]
    # Return the prediction as JSON
    result = {
        'fertilizer' : prediction
    }
    return jsonify(result)
    

#FertiBot
@app.route("/FertyBot.html", methods=["GET", "POST"])
def FertiBot():
    return render_template('FertyBot.html')

#PlantGuard
@app.route("/PlantGuard.html", methods=["GET", "POST"])
def PlanGuard():
    return render_template('PlantGuard.html')

#contact 
@app.route("/Contact.html", methods=["GET", "POST"])
def Contact():
    return render_template('Contact.html')

#help chat 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#chatbot

@app.route("/chatbot",methods=["POST","GET"])
def answer_query():
    # loading the model 
    model = joblib.load('./models/chat.pkl')
    # Get the data from the POST request
    user_input_dict = request.get_json(force=True)
    user_input = user_input_dict['input_user']
    print(user_input)
    
    
     #Make prediction
    with open('data/intents.json', 'r') as f:
        intents = json.load(f)

    File = "models/chat.pkl"
    data = joblib.load(File)

    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    model_state = data['model_state']
    tags = data['tags']

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    #running the chat
    sentence = user_input
    sentence = Tokenize(sentence)
    X = Bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent["answers"])
    else:
        response = "I’m here to assist you as a chatbot, if I can’t address your query, please contact our support team via the 'Contact Us' link for further assistance."     

    return jsonify(response)

##Fertibot
@app.route("/fertibot",methods=["POST","GET"])
def Send_Response():
    question_dict = request.get_json(force=True)
    user_question = question_dict['user_question']
    print(f"user question {user_question}")
    data_path = './data/Fertilizers'
    chroma_path = 'chroma'
    documents = load_documents(data_path)
    chunks = split_text(documents)
    #save_to_chroma(chunks, chroma_path)
    #Env Variables
    load_dotenv()
    response = query_rag(user_question,chroma_path)
    truncate_response = truncate_from_keyword(response,'Answer:')
    translate_response = detect_language(user_question,truncate_response)
    final_response = generate_final_response(translate_response)
    print(final_response)
    return jsonify(final_response)
    

##plantguard
@app.route("/plantguard",methods=["POST","GET"])
def Show_Diases():
    file = request.files['image']
    file.save(os.path.join('uploaded', 'leaf.png'))
    cnn = tf.keras.models.load_model('./models/trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img('uploaded/leaf.png',target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = cnn.predict(input_arr)
    # Import the dictionary from desease_desc.py
    result_index = np.argmax(predictions) #Return index of max element
    print(result_index)

    print(diseases[result_index])
    return jsonify(diseases[result_index])

# French

@app.route("/Home-fr.html")
def Home_fr():
    return render_template('Home-fr.html')
#ferify

@app.route("/Fertify-fr.html")
def Fertify_fr():
    return render_template('Fertify-fr.html')

#fertibot

@app.route("/FertyBot-fr.html")
def FertifyBot_fr():
    return render_template('FertyBot-fr.html')

#plantguard

@app.route("/PlantGuard-fr.html")
def PlantGuard_fr():
    return render_template('PlantGuard-fr.html')

#contact 
@app.route("/Contact-fr.html")
def Contact_fr():
    return render_template('Contact-fr.html')


    











if __name__ == '__main__':
    app.run(debug=True)
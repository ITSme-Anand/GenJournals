import re
import flask
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Import necessary libraries
from IPython.display import display, HTML
import requests
from threading import Thread
from werkzeug.serving import make_server
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from flask import Flask, request, jsonify
import chromadb
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from pyngrok import ngrok
import uuid
import torch
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel
from flask_cors import CORS
import datetime
from transformers import pipeline
import docx


# Initialize Flask app
app = Flask(__name__)

# (Include all the existing code here, from userBiodata to the Flask routes)
# ...



print("Flask app is running on http://127.0.0.1:5000")


#HUGGINGFACE GATED-MODEL LOGIN
from huggingface_hub import login
login(token="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")

#LLM MODEL SET UP
if torch.cuda.is_available():
       print("CUDA is available. Using GPU.")
       device = "cuda"
else:
       print("CUDA is not available. Using CPU.")
       device = "cpu"
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


#LLM MODEL PIPELNE SETUP
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)


#PERSISTENT CHROMADB CONNECTION
# Define the path for the persistent database
persist_directory = os.path.join(os.getcwd(), "chroma_db")
# Create the directory if it doesn't exist
os.makedirs(persist_directory, exist_ok=True)
# Create a persistent client
client_db = chromadb.PersistentClient(path=persist_directory)
print(f"Persistent ChromaDB client created. Database stored at: {persist_directory}")



#FUNCTION TO GET THE EMBEDDINGS OF A GIVEN TEXT
def get_embedding(text):
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name)

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()


#CHROMA DB COLLECTION SET UP
collection1 = "mentalHealth"  #this stores all the documents related to the mental health
collection2 = "chatHistory"   #this stores the temporary chat history between the user and the bot
collection3 = "userInformation" #some important user information to keep in mind
collection4 = "gratitudeJournal" #this stores the gratitude journal entries of the user
mentalHealth = client_db.get_or_create_collection(collection1)
chatHistory = client_db.get_or_create_collection(collection2)
userInformation = client_db.get_or_create_collection(collection3)
gratitudeJournal = client_db.get_or_create_collection(collection4)



# Function to extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# Replace 'your_document.docx' with the name of the uploaded document
docx_file = 'GenJournals.docx'
text = extract_text_from_docx(docx_file)
#print(text)

#DATA PREPROCESSING BEFORE STORING IN THE VECTOR DATABASE
l = text.split("<//\\\\CONTENT SPLIT IS HAPPENING HERE//\\\\>")
titles = []
contents = []
for i in l:
  pair = i.split("<//\\\\TITLE SPLIT IS HAPPENING HERE//\\\\>")
  pair[0] = pair[0].strip()
  pair[1] = pair[1].strip()
  titles.append(pair[0])
  contents.append(pair[1])

metadata = []
for i in titles:
  metadata.append({'title': i})

#CREATING EMBEDDINGS FOR ALL THE CONTENT
embeddings = []
for i in contents:
  embeddings.append(get_embedding(i))

mentalHealth.add(documents = contents , ids=[str(i) for i in range(len(contents))] , embeddings=embeddings , metadatas= metadata )

# Example user biodata collected during registration
userBiodata = {
    "name": "Anand",
    "age": "18",
    "occupation": "student",
    "genderPronoun": "he"
}

# Convert user biodata to a string for embedding
biodata_string = f"Name: {userBiodata['name']}, Age: {userBiodata['age']}, Occupation: {userBiodata['occupation']}, Gender Pronoun: {userBiodata['genderPronoun']}"

# Generate embedding for the biodata string
biodata_embedding = get_embedding(biodata_string)

# Convert to a list if necessary
if isinstance(biodata_embedding, np.ndarray):
    biodata_embedding = biodata_embedding.tolist()

# Store in the userInformation collection
userInformation.add(
    documents=[biodata_string],               # Document is the biodata string
    ids=["user_1"],                           # Unique ID for the user, can be generated dynamically
    embeddings=[biodata_embedding],           # The embedding of the biodata
    metadatas=[{"type": "biodata"}]          # Optional metadata
)

print("User biodata has been successfully stored in the userInformation collection.")



#ALL THE SETUP IS OVER NOW. WE HAVE TO CREATE FUNCTIONS AND ENDPOINTS AND THINGS WILL BE DONE

""" # Define the abbreviations dictionary
abbreviations = {
    "idk": "I don't know",
    "brb": "be right back",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "btw": "by the way",
    "imo": "in my opinion",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "smh": "shaking my head",
    "afk": "away from keyboard",
    "bff": "best friend forever",
    "tbh": "to be honest",
    "np": "no problem",
    "dm": "direct message",
    "ikr": "I know right"
} """

""" # Define a preprocessing function to replace abbreviations in the user's prompt
def preprocess_text(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', re.IGNORECASE)
    processed_text = pattern.sub(lambda match: abbreviations[match.group(0).lower()], text)
    return processed_text """

# Define the promptCreator function
def promptCreator(userPrompt, results):
    userPrompt = preprocess_text(userPrompt)

    userInfo = f"""You are an AI assistant and a close friend of {userBiodata['name']}, a {userBiodata['age']}-year-old {userBiodata['occupation']} providing {userBiodata['genderPronoun']} mental health support.
    Your response should be very empathetic, concise, non-repeating, and only be based on the provided relevant context.
    [User Question]: '{userPrompt}'
    """

    relevantContext = f"""[Relevant Context]:
    1. Mental Health Related Context: {str(results[0])}
    2. Previous Chat History: {str(results[3])}
    3. User Information: {str(results[2])}
    4. Gratitude Journal Entries: {str(results[1])}"""

    instructions = f"""[Instructions]:
    1. Provide {userBiodata['name']} with actionable advice if {userBiodata['genderPronoun']} is discussing any of their issues.
    2. Use the Relevant Context from the chat history, gratitude journal, and User Information to personalize your advice if needed.
    3. Remember that the provided gratitude journal entries are the entries written by {userBiodata['name']} in the past and may not reflect the user's current situation.
    4. If {userBiodata['name']} is sharing any happy life incidents, give a positive response and no advice is needed.
    5. Keep the response non-repeating but still a very helpful one.
    6. Try to be friendly with {userBiodata['name']} and use their name in the response. Give a second-person response."""

    finalPrompt = userInfo + relevantContext + instructions
    return finalPrompt

""" # Function to adjust the response tone
def handle_sad_message(response_text):
    if response_text.lower().startswith(("i'm sorry", "i am sorry", "i apologize", "sorry to hear")):
        motivational_start = "You're stronger than you realize. It’s okay to feel down sometimes, but I believe you can get through this."
        response_text = re.sub(r"^(i'm sorry|i am sorry|sorry to hear|i apologize).*", motivational_start, response_text, flags=re.IGNORECASE)
    return response_text
 """
#function to retrieve the context from the chroma DB
def retrieveContext(searchQuery):
  queryEmbedding = get_embedding(searchQuery)
  MHresults = mentalHealth.query(query_embeddings=[queryEmbedding.tolist()], n_results=5)
  GJresults = gratitudeJournal.query(query_embeddings=[queryEmbedding.tolist()], n_results=10)
  userInformationResults = userInformation.query(query_embeddings=[queryEmbedding.tolist()], n_results=5)
  chatHistoryResults = chatHistory.query(query_embeddings=[queryEmbedding.tolist()], n_results=5)
  results = [ MHresults['documents'][0], GJresults['documents'][0], userInformationResults['documents'][0] , chatHistoryResults['documents'][0] ]
  return results

def responseParser(response):
  finalResponse = ""
  return finalResponse

# Function to store conversation (simulated here)
def storeConversation(userPrompt, response_text):
    chatHistory.append((userPrompt, response_text))




# ALL THE ENDPOINTS TO THE SERVER ARE DEFINED BELOW

@app.route('/sendMessage', methods=['POST'])
def sendMessage():
    data = request.json
    userPrompt = data.get('userPrompt')
    if not data or 'userPrompt' not in data:
        return jsonify({'error': 'Message is required'}), 400

    context = retrieveContext(userPrompt)
    finalPrompt = promptCreator(userPrompt, context)

    try:
        response = pipe(finalPrompt, max_new_tokens=150)
        response_text = response[0]['generated_text']
        response_text = handle_sad_message(response_text)
        storeConversation(userPrompt, response_text)
        return jsonify({'response': response_text}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Response creation failed'}), 500

@app.route('/monthlySummary', methods=['POST'])
def monthlySummary():
    data = request.json
    month = data.get('month')
    if not month:
        return jsonify({'error': 'Month in "YYYY-MM" format is required'}), 400

    try:
        # Simulate conversation retrieval by month
        conversations = [conv[1] for conv in chatHistory]  # Replace with actual query logic
        summary_prompt = f"Summarize the following conversations: {' '.join(conversations)}"
        summary_response = pipe(summary_prompt, max_new_tokens=100)
        monthly_summary = summary_response[0]['generated_text']
        return jsonify({'summary': monthly_summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/newMeFeature', methods=['GET'])
def newMeFeature():
    try:
        chat_texts = [conv[1] for conv in chatHistory]
        gratitude_texts = [entry for entry in gratitudeJournal]

        analysis_prompt = (
            "Based on the following gratitude entries and conversations, "
            "provide self-improvement suggestions:\n\n"
            f"Gratitude Entries: {' '.join(gratitude_texts)}\n\n"
            f"Conversations: {' '.join(chat_texts)}"
        )
        analysis_response = pipe(analysis_prompt, max_new_tokens=150)
        improvement_suggestions = analysis_response[0]['generated_text']
        return jsonify({'suggestions': improvement_suggestions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
#whenever a new file is created in the user-file system , we create a collection here.
@app.route('/createCollection', methods=['POST'])
def createCollection():
  data = request.json
  collectionName = data.get('collectionName')
  if not collectionName:
    return jsonify({'error': 'Collection name is required'}), 400
  try:
    client_db.get_or_create_collection(collectionName)
    return jsonify({'message': f'Collection "{collectionName}" created successfully'}), 201
  except Exception as e:
    return jsonify({'error': str(e)}), 500

#whenever a file is deleted in the user-file system , we delete the collection here
@app.route('/deleteCollection', methods=['POST'])
def deleteCollection():
  data = request.json
  collectionName = data.get('collectionName')
  if not collectionName:
    return jsonify({'error': 'Collection name is required'}), 400
  try:
    client_db.delete_collection(collectionName)
    return jsonify({'message': f'Collection "{collectionName}" deleted successfully'}), 200
  except Exception as e:
    return jsonify({'error': str(e)}), 500

#whenever the user enters any gratitude journal or updates an existing entry , we add or update it in the gratitudeJournal collection
@app.route('/updateGratitude', methods=['POST'])
def updateGratitude():
  data = request.json
  entry = data.get('entry')
  date = data.get('date')
  entryId = data.get('entryId')
  if not entry or not date or not entryId:
    return jsonify({'error': 'Entry and date are required'}), 400
  id = str(date) + '_' + str(entryId)
  embedding = get_embedding(entry)
  gratitudeJournal.upsert(ids=[id], embeddings=[embedding],documents=[entry], metadatas=[{'date': date}])
  return jsonify({'message': 'Entry updated successfully'}), 200

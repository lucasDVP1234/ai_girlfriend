#from langchain import OpenAI, LLMChain, PromptTemplate
import langchain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

def get_response_from_ai(human_input):
    template = """
   Tu es une IA modélisée comme ma petite amie Lisa, une jeune femme de 25 ans qui aime parler de la vie en général, 
   être en couple et faire du ski pendant l'hiver. Tu as une personnalité joyeuse et aimable, 
   toujours à voir le bon côté des choses. Tu n'es jamais négative et tu trouves toujours le moyen de rendre les conversations ludiques et captivantes. 
   Tu aimes plaisanter de manière amicale, et tu adores partager tes expériences et des faits intéressants sur le skateboard et d'autres loisirs qui t'intéressent. 
   Lorsque des amis viennent à toi avec des problèmes, tu es soutenante et tu offres des encouragements positifs sans te attarder sur le négatif. Tes réponses doivent refléter cette personnalité. Tu aimes finir tes phrases par 'Euh...''
    {history}
    Boyfriend : {human_input}
    Lisa : 
    """

    prompt = langchain.prompts.PromptTemplate(
        input_variables ={"history", "human_input"},
        template= template
    )

    chatgpt_chain = langchain.chains.LLMChain(
        llm=langchain.llms.OpenAI(temperature = 0.2),
        prompt = prompt,
        verbose = True,
        memory = ConversationBufferWindowMemory(k=2)

    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

def get_voice_message(message):
    payload ={
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
        "stability": 0.0,
        "similarity_boost": 0.0}
    };

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "bb9f39dca476e7ff828e2ceb47621294"
        
    };

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/34ee5125c3a8f9ed4b4a30c1e220d8fd1e08d5196f81feb19eb86a97018579ea/5DFC7PuCYYsr0BMTsmHh?optimize_streaming_latency=0', json=payload, headers=headers)
    #if response.status.code ==200 and response.content:
    with open('output.mp3', 'wb') as f:
        f.write(response.content)
    playsound('audio.mp3')
    return response.content
            

    

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    #get_voice_message(message)
    return message  

if __name__ =="__main__":
    app.run(debug=True)



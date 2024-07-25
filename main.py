# working code
import os
import json
from fastapi import FastAPI, Depends, HTTPException
from starlette.config import Config
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import HTMLResponse, RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel
import requests
from datetime import datetime, timezone
from googleapiclient.errors import HttpError
from starlette.middleware.cors import CORSMiddleware
from typing import List, Optional

#-+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma
import os

# google genrative embedding model api
os.environ["GOOGLE_CLIENT_ID"] = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
embedding  = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

#-+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-
def parse_workshop_details(workshop_str):
    workshop_parts = workshop_str.split()
    title = workshop_parts[0]
    start_time, end_time = workshop_parts[1:]
    return title, start_time, end_time

def schedule_event(calendar_id, service, title, start_datetime, end_datetime):
    event = {'summary': title,'start': {'dateTime': start_datetime,},'end': {'dateTime': end_datetime,}}
    service.events().insert(calendarId=calendar_id, body=event).execute()
    response = "Event "+title+" created successfully!"
    return response

def sch(question): 
    prompt = PromptTemplate.from_template('''
                    "you are a ai who does the nemed entity recongintion and convert it to the given format"
                    "example quetion: schedula a event named meeting on the 4 july at 10 am for one hour"
                    "format: event-name start-time end-time"
                    "answer format: Meeting 2024-04-04T10:00:00+05:30 2024-04-04T11:00:00+05:30 "
                    "convert this: {question}" 
                    "separte this three values with space"
                    strictly follow the answer format:  event name without space use _ in place of space 2024-MM-DDTHH:MM:SS+05:30 2024-MM-DDTHH:MM:SS+05:30''')
    messages = prompt.format(question=question)
    result = llm.invoke(messages)
    return result
def mainChat(message, token, filename):
    string = message
    creds = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=['https://www.googleapis.com/auth/calendar']
        )
    if string.startswith("/sch"):
        service = build('calendar', 'v3', credentials=creds)
        calendar_id = 'primary'
        result = sch(message)
        title, start_datetime, end_datetime = parse_workshop_details(result.content)
        response = schedule_event(calendar_id, service, title, start_datetime, end_datetime)
    else:
        response = ragc(message, token, filename)
    return response
#-+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-


def ragc(message, token, filename):
    data = data_ingestion(token)
    if data == "null" or not data:
        return "There are no events in the calendar. Please check your calendar or try logging in again.."
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, f"{filename}.txt")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Existing data in '{file_path}' has been deleted.")
        with open(file_path, 'w') as file:
            file.write(data)
        print(f"Data written successfully to {file_path}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(docs, embedding)
    retriever = vectordb.as_retriever()
    template = template = """you are an ai calendar assistant you help user to optimized and to view their schedule with simple language by using above context  :
    {context}
    the time gaps which is not shown in the context is unscheduled or the free time for the user with all this information you have to assist the user politely
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = llm
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    response = chain.invoke(message)
    return response
#-+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-
# calender data ingestion from the primiary calendarId
def data_ingestion(token):
    creds = Credentials(
    token=token['access_token'],
    refresh_token=token.get('refresh_token'),
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scopes=['https://www.googleapis.com/auth/calendar']
    )
    try:
        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=20,
                singleEvents=True,
                orderBy="startTime",).execute())
        events = events_result.get("items", [])
        if not events:
            print("No upcoming events found.")
            data = "No upcoming events found."
            return
        event_data = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            event_data.append({
                "start": start,
                "end": end,
                "summary": event["summary"]
            })
        
        datal = event_data
        print(datal)
        data = str(datal)
        '''
        filename = "./data/data.txt"
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Existing data in '{filename}' has been deleted.")
            with open(filename, 'w') as file:
                file.write(data)
            print(f"Data written successfully to {filename}")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        '''   
    except HttpError as error:
        data = "null"
        print(f"An error occurred: {error}")
    return data

#-+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="!secret")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "https://ajinkyavbhandare.github.io/website", "https://ajinkyavbhandare.github.io"],  # Note: changed from 127.0.0.1 to localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

os.environ["GOOGLE_CLIENT_ID"] = os.environ.get('GOOGLE_CLIENT_ID')
os.environ["GOOGLE_CLIENT_SECRET"] = os.environ.get('GOOGLE_CLIENT_SECRET')

config = Config()
oauth = OAuth(config)

CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
oauth.register(
    name='google',
    server_metadata_url=CONF_URL,
    client_kwargs={
        'scope': 'https://www.googleapis.com/auth/calendar openid email profile'
    }
)


@app.get('/')
async def homepage(request: Request):
    user = request.session.get('user')
    token = request.session.get('token')
    if user:
        data = json.dumps({"user": user, "token": token}, indent=2)
        html = (
            f'<pre>{data}</pre>'
            '<a href="/logout">logout</a><br>'
            '<a href="/cal">Google Calendar</a>'
        )
        return HTMLResponse(html)
    return HTMLResponse('<a href="/login">login</a>')


@app.get('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    request.session.clear() # Clear the session before starting the login flow
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        raise HTTPException(status_code=400, detail=f"OAuth Error: {error.error}")
    user = token.get('userinfo')
    if user:
        request.session['user'] = dict(user)
        request.session['email'] = user['email']
        request.session['token'] = token
    return HTMLResponse(content=f"""
        <script>
            window.opener.postMessage({json.dumps(token)}, '*');
            window.close();
        </script>
    """)


@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    request.session.clear() # Clear the session on logout
    return RedirectResponse(url='/')

class InputText(BaseModel):
    text: str

@app.post("/echo")
async def echo_text(input_text: InputText, request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="You need to be logged in to chat.")
    response = query({"inputs": input_text.text})
    return {"text": response}

@app.post("/chat")
async def chat(input_text: InputText, request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="You need to be logged in to chat.")
    response = mainChat(input_text.text, token=request.session.get('token'), filename=request.session.get('email'))
    return {"text": response}
    

@app.get('/cal')
async def cal(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="You need to be logged in to access this page.")
    calendar_list = data_ingestion(request.session.get('token'))
    return HTMLResponse(f'<pre>{json.dumps(calendar_list, indent=2)}</pre><br><a href="/">Home</a>')

class UserInfo(BaseModel):
    iss: str
    azp: str
    aud: str
    sub: str
    email: str
    email_verified: bool
    at_hash: str
    nonce: str
    name: str
    picture: str
    given_name: str
    family_name: str
    iat: int
    exp: int

class TokenPayload(BaseModel):
    access_token: str
    expires_in: int
    scope: str
    token_type: str
    id_token: str
    expires_at: int
    userinfo: UserInfo

@app.post("/token-dict")
async def receive_token_dict(token_payload: TokenPayload):
    try:
        token = token_payload.dict()
        response = data_ingestion(token)
        return response
    except Exception as e:
        print(f"Error processing token: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing token: {str(e)}")

@app.post("/mainapi")
async def chat(token_payload: TokenPayload, input_text: InputText):
    token = token_payload.dict()
    filename = token['userinfo']['email']
    response = mainChat(input_text.text, token, filename)
    return {"text": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

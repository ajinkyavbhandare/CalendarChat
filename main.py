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


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="!secret")

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
    if user:
        data = json.dumps(user)
        html = (
            f'<pre>{data}</pre>'
            '<a href="/logout">logout</a>'
        )
        return HTMLResponse(html)
    return HTMLResponse('<a href="/login">login</a>')


@app.get('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        return HTMLResponse(f'<h1>{error.error}</h1>')
    user = token.get('userinfo')
    if user:
        request.session['user'] = dict(user)
        request.session['token'] = token
    return RedirectResponse(url='/')


@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
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


@app.get('/cal')
async def cal(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="You need to be logged in to access this page.")
    
    token = request.session.get('token')
    if not token:
        raise HTTPException(status_code=401, detail="Token not found.")
    
    # Convert token to credentials
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=['https://www.googleapis.com/auth/calendar']
    )
    
    # Build the calendar service
    service = build("calendar", "v3", credentials=credentials)
    
    # Example of retrieving the list of calendars
    calendar_list = service.calendarList().list().execute()
    return HTMLResponse(f'<pre>{json.dumps(calendar_list, indent=2)}</pre><br><a href="/">Home</a>')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

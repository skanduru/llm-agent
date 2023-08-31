
import os
import openai
from fastapi import FastAPI, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.routing import request_response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv

from movieRec import createAgent, shutdown_index, runAgent
from AppLogger import get_logger

load_dotenv(find_dotenv())

app = FastAPI()
templates = Jinja2Templates(directory="templates/")  # Assuming you have a templates directory

app.mount("/static", StaticFiles(directory="static"), name="staticfiles")

openai.api_key = os.getenv("OPENAI_API_KEY")
logger = get_logger("index")

@app.on_event("startup")
async def startup_event():
    """
    This function will be called when the application starts.
    It's responsible for setting up the database and AI chat agents and
    any other application settings
    """
    logger.info("Starting up...")
    createAgent()  # Initializing agent

@app.on_event("shutdown")
async def shutdown_event():
    """
    This function will be called when the application shuts down.
    It's responsible for closing the database session.
    """
    logger.info("Shutting down...")
    shutdown_index()

@app.get("/")
def read_root(request: Request, result: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/generate")
def generate(request: Request, userInput: str = Form(...), number: str = Form(...)):
    data = generate_prompt(userInput, number)
    return RedirectResponse(url=f"/?result={data}", status_code=302) # Change this line

def generate_prompt(userInput, number):
    genre, Rec_movies = runAgent(userInput, int(number))
    logger.info(f"Received inputs: String: \"{userInput}\", number: {number}. genre: {genre} deciphered")
    return """Recommend the best movies to watch in genre: {} cap: {}\n{}""".format(genre, number, Rec_movies)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.model import load_model, generate_response

# Инициализация приложения
app = FastAPI()

# Загрузка модели
tokenizer, model = load_model()

# Шаблоны для фронтенда
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": ""})


@app.post("/", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    prompt = f"Persona A: {user_input}\nPersona B:"
    response = generate_response(prompt, tokenizer, model)
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "user_input": user_input})

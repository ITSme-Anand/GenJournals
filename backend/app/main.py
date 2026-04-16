from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='GenJournals Backend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials = True,
    allow_methods= ["*"],
    allow_headers = ["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def return_health_status():
    return {"app working well!"}

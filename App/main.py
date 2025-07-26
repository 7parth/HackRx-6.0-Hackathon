from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from App.routers import documents, queries
from App.database import engine, SessionLocal, Base



Base.metadata.create_all(bind=engine) # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(documents.router)
app.include_router(queries.router)



@app.get("/")
def root():
    return {"message": "API is working!"}
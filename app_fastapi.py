from fastapi import FastAPI
from pydantic import BaseModel
from agent import agent
import logging

logging.info('-------- hello I have just started the app :) ---------')

# 5. Set this as an API endpoint via FastAPI
app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content

import functools
from typing import Optional

from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()


@functools.lru_cache(maxsize=1)
def get_pipe(task: Optional[str] = None):
    return pipeline(task)


@app.get('/')
def return_version():
    with open('requirements.txt') as f:
        return f.read()


@app.post('/pipeline/{task}')
async def get_result(task: str, payload: Request):
    payload = await payload.json()
    try:
        pipe = get_pipe(task)
        result = pipe(payload)
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}

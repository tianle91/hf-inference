import functools
from typing import Optional

from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()


@functools.lru_cache(maxsize=1)
def get_pipe(task: Optional[str] = None):
    return pipeline(task)


@app.post('/pipeline')
async def get_result(payload: Request):
    payload = await payload.json()
    try:
        pipe = get_pipe(payload['task'])
        result = pipe(payload['input'])
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}

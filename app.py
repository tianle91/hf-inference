import functools
import os

from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()

NUM_CACHED_PIPELINES = os.getenv('NUM_CACHED_PIPELINES', '')
if len(NUM_CACHED_PIPELINES) == 0:
    NUM_CACHED_PIPELINES = 1
else:
    NUM_CACHED_PIPELINES = int(NUM_CACHED_PIPELINES)

with open('requirements.txt') as f:
    REQUIREMENTS_TXT = f.read()


@functools.lru_cache(maxsize=NUM_CACHED_PIPELINES)
def get_pipeline(**kwargs):
    return pipeline(**kwargs)


@app.get('/')
def return_version():
    return REQUIREMENTS_TXT


@app.post('/pipeline')
async def get_result(payload: Request):
    payload = await payload.json()
    params = payload['params']
    input = payload['input']
    try:
        pipe = get_pipeline(**params)
        result = pipe(input)
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}

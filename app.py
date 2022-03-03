import functools

from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()


@functools.lru_cache(maxsize=1)
def get_pipeline(**kwargs):
    return pipeline(**kwargs)


@app.get('/')
def return_version():
    with open('requirements.txt') as f:
        return f.read()


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

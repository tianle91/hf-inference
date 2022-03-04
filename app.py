import functools
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_LMHEAD = GPT2LMHeadModel.from_pretrained('gpt2')

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


@app.get('/', response_class=HTMLResponse)
def return_version():
    return HTMLResponse(content=f'''
    <html>
        <head>
            <title>HF Inference</title>
        </head>
        <body>
            See
            <a href=https://github.com/tianle91/hf-inference>tianle91/hf-inference</a>
            for more information.
            <h1>Requirements.txt</h1>
            {REQUIREMENTS_TXT}
        </body>
    </html>
    ''', status_code=200)


@app.post('/pipeline')
async def get_pipeline_result(payload: Request):
    payload = await payload.json()
    params = payload['params']
    input = payload['input']
    try:
        pipe = get_pipeline(**params)
        result = pipe(input)
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}


@app.post('/gpt2loss')
async def get_gpt2_loss(payload: Request):
    payload = await payload.json()
    try:
        inputs = GPT2_TOKENIZER(payload, return_tensors="pt")
        outputs = GPT2_LMHEAD(**inputs, labels=inputs["input_ids"])
        return {'loss': str(float(outputs.loss))}
    except Exception as e:
        return {'error': str(e)}

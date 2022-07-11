from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = FastAPI()

with open('requirements.txt') as f:
    REQUIREMENTS_TXT = f.read()

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
        pipe = pipeline(**params)
        result = pipe(input)
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}


@app.post('/gpt2loss')
async def get_gpt2_loss(payload: Request):
    payload = await payload.json()
    try:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        lmhead = GPT2LMHeadModel.from_pretrained('gpt2')
        inputs = tok(payload, return_tensors="pt")
        outputs = lmhead(**inputs, labels=inputs["input_ids"])
        return {'loss': str(float(outputs.loss))}
    except Exception as e:
        return {'error': str(e)}

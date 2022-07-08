# HF Inference
I encountered so many `unknown_errors` when querying the huggingface api so I'm going to try making my own.

## How to Use
```
docker run --platform linux/amd64 -p 8000:8000 tianlechen/hf-inference
```
A gpu can improve performance significantly, you can add the `--gpus all` parameter to docker run if you have one set up.
If you have Windows 11, it should be [fairly simple](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/)
You can also just use the existing `docker-compose.yaml`.
```
docker-compose up
```

## What it does
The following should work (the first query might take a while, but subsequent ones should be much faster):
```python
>>> import requests
>>> payload = {'input': 'The goal of life is <mask>.', 'params': {'task': 'fill-mask'}}
>>> requests.post('http://localhost:8000/pipeline', json=payload).json()
{'result': [{'score': 0.06897158920764923,
             'sequence': 'The goal of life is happiness.',
             'token': 11098,
             'token_str': ' happiness'}, ...]}
```

This endpoint just wraps around [`pipeline`](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline).
Let's break down the payload.
```python
payload = {
    # argument to a created pipeline object
    'input': 'The goal of life is <mask>.', 
    # parameters passed to create a pipeline
    'params': {'task': 'fill-mask'}
}
```
These two expressions should yield the same results (with the main advantage being that the computation is running somewhere else).
```python
>>> requests.post('http://localhost:8000/pipeline', json=payload).json()['result']
...
>>> pipeline(**payload['params'])(payload['input'])
...
```
For details on what parameters (for example, what other tasks you can run), check out [the documentation on Hugging Face](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline.task)


# Another endpoint
Because it's not part of a standard pipeline and I need it.
It just computes the loss for GPT-2.
```python
>>> payload ='testing a longer sentence'
>>> requests.post('http://localhost:8000/gpt2loss', json=payload).json()
{'loss': '6.174161434173584'}
```

# Useful

## Docker stuff
```
docker build -t tianlechen/hf-inference .
docker push tianlechen/hf-inference
```

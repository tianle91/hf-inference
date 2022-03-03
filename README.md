# HF Inference
I encountered so many `unknown_errors` when querying the huggingface api so I'm going to try making my own.

Just run `docker-compose up` (you'll need [docker](https://www.docker.com/products/docker-desktop) installed).
The following should work (the first query might take a while, but subsequent ones should be much faster):
```python
>>> import requests
>>> payload = {'input': 'The goal of life is <mask>.', 'params': {'task': 'fill-mask'}}
>>> requests.post('http://localhost:8000/pipeline', json=payload).json()
{'result': [{'score': 0.06897158920764923,
             'sequence': 'The goal of life is happiness.',
             'token': 11098,
             'token_str': ' happiness'},
            ...
            {'score': 0.02376789040863514,
             'sequence': 'The goal of life is simplicity.',
             'token': 25342,
             'token_str': ' simplicity'}]}
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


## GPU
The following uses a gpu `docker-compose -f docker-compose-gpu.yaml up`.
Make sure you have it set up for your host machine.
If you have Windows 11, it should be [fairly simple](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/), which is what I've tested this on.


## ⚠️⚠️ Caution ⚠️⚠️
There's no security - this is intended to be deployed on a local private network.

## Caching
Keeping more `pipeline` objects in memory might improve performance.
This is configured with the `NUM_CACHED_PIPELINES` environment variable.
If your `.env` file looks like the following then there will be `10` pipelines being cached.
```
cat .env
NUM_CACHED_PIPELINES=10
```

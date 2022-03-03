# HF Inference
I encountered so many `unknown_errors` when querying the huggingface api so I'm going to try making my own.

Just run `docker-compose up` (you'll need [docker](https://www.docker.com/products/docker-desktop) installed).
The following should work:
```python
>>> import requests
>>> payload = {'input': 'The goal of life is <mask>.', 'params': {'task': 'fill-mask'}}
>>> requests.post('http://localhost:8000/pipeline', json=payload).json()
{'result': [{'score': 0.06897158920764923, 'token': 11098, 'token_str': ' happiness', 'sequence': 'The goal of life is happiness.'}, {'score': 0.06554900109767914, 'token': 45075, 'token_str': ' immortality', 'sequence': 'The goal of life is immortality.'}, {'score': 0.03235733136534691, 'token': 14314, 'token_str': ' yours', 'sequence': 'The goal of life is yours.'}, {'score': 0.024313855916261673, 'token': 22211, 'token_str': ' liberation', 'sequence': 'The goal of life is liberation.'}, {'score': 0.02376789040863514, 'token': 25342, 'token_str': ' simplicity', 'sequence': 'The goal of life is simplicity.'}]}
```
The queries should be much faster after the first one.


## ⚠️⚠️ Caution ⚠️⚠️
There's no security - this is intended to be deployed on a local private network.


## Details
The payload json is just a wrapper around `pipeline`.
For details, see https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline

Let's break down the payload.
```python
payload = {
    'input': 'The goal of life is <mask>.', 
    'params': {'task': 'fill-mask'}
}
```
- `input`: Inputs passed into an instantiated `pipeline`.
- `params`: Parameters passed into a `pipeline`.

These two statements should be equivalent.
```python
>>> requests.post('http://localhost:8000/pipeline', json=payload).json()['result']
```
```python
>>> pipeline(**payload['params'])(payload['input'])
```


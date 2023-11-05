# Simple Local LLM server

## What is this ?

This project is mostly a personal utility that I use and that finally decided to share. 

This is a little server that runs a local version of a quantized Mistral 7B Instruct in GGUF format (thank you theBloke), as well as sentence transformer's all-MiniLM-L6-v2 for computing encodings.

The reason why I made this is that those models take a long time to load into memory, and thus it is more efficient to have a local server making those models available all the time instead of reloading them each time I want to use them in a script. And also I didn't know VLLM was a thing 😂.

## How to start the server 

1. Make sure you have all of the requirements

``` sh
pip install -r requirements.txt
```
2. Make sure you download the encoding model locally too. The script get_encoder.py was made for that:

``` sh
cd server
python get_encoder.py
```
3. Make sure you have uvicorn installed

``` sh 
pip install uvicorn
```

4. Run the server! 

```sh 
python -m uvicorn main:app
```


## How to use the client

### Text Generation

```python

from llm_client import LocalLLMClient as LLM

# initialise the client with the local llm url
llm = LLM("http://127.0.0.1:8000")


# get a whole response all at once
funny_joke : str = llm("Tell me a funny joke about cats", temperature = 0, stop = ["HAHAHAHA"])

# stream the response from the server instead and accessing the tokens as a generator object
for token in llm("How long does it take to learn AI?", stream = True):
    print(token, end = "", flush = True)


```


### Embeddings

```python 

from llm_client import LocalEmbeddingsClient
import numpy as np

# initialise the encoder client with the local server url 
encoder = LocalEmbeddingsClient("http://127.0.0.1:8000")


sentences = ["Brazil has 5 world cup titles, hell yeah", "I like soccer", "Make love not war"]

# you just have to pass a list of senteces to the encoder
encoddings : np.ndarray = encoder.encode(sentences)

# you can also pass only 1 string
one_encodding : np.ndarray = encoder.encode("Maybe give me a star °^° ?")

```
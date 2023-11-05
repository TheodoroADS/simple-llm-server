from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from ctransformers import AutoModelForCausalLM as LLM
from sentence_transformers import SentenceTransformer
from os.path import join
import requests

def is_internet_available():

    try:
        requests.head("http://www.google.com")
        return True
    except requests.ConnectionError:
        return False


llm_name = r"TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
encoder_local_path = join(".", "all-MiniLM-L6-v2")

# this method has the annoying issue that it raises an exception if local_files_only is not set to False and there is no internet
llm = LLM.from_pretrained(llm_name, context_length = 4096, local_files_only=is_internet_available())
encoder = SentenceTransformer(encoder_local_path)
    
app = FastAPI()

class GenerationArgs(BaseModel):

    prompt : str
    temperature : float = 0
    stop : List[str] | None = None

class EncodingArgs(BaseModel):

    sentences : List[str]

@app.get("/ping/")
async def ping():

    '''
    This endpoints can be used to check the connection to the server
    '''

    return

@app.post("/prompt/")
async def answer(args : GenerationArgs):

    '''
    This endpoint will synchronously compute the LLM's reponse to the user prompt and return it all at once

    :param prompt : str -> the prompt for the LLM
    :param temperature : float -> the temperature applied. Default is 1.
    :param stop : List[str] | None -> the list of tokens that will stop generation once they are generated. Default is None

    '''

    response = llm(args.prompt, temperature= args.temperature, stop = args.stop)

    return {"message" : response}

@app.post("/prompt-streaming")
async def answer_streaming(args : GenerationArgs):

    '''
    This endpoint will compute the LLM's reponse to the user prompt, streaming the tokens as they are generated to the user

    :param prompt : str -> the prompt for the LLM
    :param temperature : float -> the temperature applied. Default is 1.
    :param stop : List[str] | None -> the list of tokens that will stop generation once they are generated. Default is None

    '''

    response_stream = llm(args.prompt, temperature= args.temperature, stop = args.stop, stream= True)

    return StreamingResponse(response_stream)


@app.post("/encode")
async def encode(args : EncodingArgs):

    '''
    This endpoint is for calculating the embeddings (encoding) of a list of sentences 

    :param sentences : List[str] -> The list of sentences to be encoded

    '''

    return {"embeddings" : encoder.encode(args.sentences, convert_to_numpy=True).tolist()}

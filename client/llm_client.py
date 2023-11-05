import requests
from typing import List
import numpy as np


def raise_if_http_error(response : requests.Response) -> None:

    if response.status_code >= 300:
        raise requests.HTTPError(
            f"Server returned status code {response.status_code}. Stated reason is : {response.reason}"
        )


class LocalLLMClient:

    def __init__(self, url : str, prompt_template = "{prompt}", verbose : bool = True):

        self.url = url
        self.prompt_template = prompt_template
        self.verbose = verbose

        # pinging the server to check connection
        res = requests.get(url + "/ping")
        if res.status_code >= 300:
            raise requests.HTTPError(f"Could not connect to server. Status code = {res.status_code}, reason = {res.reason}")
        
        if self.verbose:
            print("Connected to server!")

    def prompt_request(self, prompt : str, temperature : float = 1, stop : List[str] | None = None ):

        payload = {
            "prompt" : self.prompt_template.format(prompt = prompt), 
            "temperature" : temperature, 
            "stop" : stop
        }

        response = requests.post(self.url + "/prompt/", json = payload)

        raise_if_http_error(response)

        return response.json()["message"]
    
    def streaming_prompt_request(self, prompt : str, temperature : float = 1, stop : List[str] | None = None ):

        payload = {
            "prompt" : self.prompt_template.format(prompt = prompt), 
            "temperature" : temperature, 
            "stop" : stop
        }

        with requests.Session() as session:
            with session.post(self.url + "/prompt-streaming", json = payload, stream = True) as resp:
                token : bytes
                for token in resp.iter_content(None):
                    if token:
                        yield token.decode('utf-8')


    def __call__(self, prompt : str, temperature : float = 1, stop : List[str] | None = None, stream = False):

        if stream:
            return self.streaming_prompt_request(prompt, temperature, stop)
        else:
           return self.prompt_request(prompt, temperature, stop)


class LocalEmbeddingsClient:

    def __init__(self, url : str, verbose : bool = True):
        
        self.url = url
        self.verbose = verbose

        # pinging the server to check connection
        res = requests.get(url + "/ping")
        if res.status_code >= 300:
            raise requests.HTTPError(f"Could not connect to server. Status code = {res.status_code}, reason = {res.reason}")
        
        if self.verbose:
            print("Connected to server!")

    def encode(self, sentences : str | List[str]) -> np.ndarray:

        if isinstance(sentences, str):
            sentences = [sentences]
        elif not isinstance(sentences, list) or any(not isinstance(sentence, str) for sentence in sentences):
            raise ValueError("The sentences argument must be a string or a list of strings")
        
        payload = {"sentences" : sentences}

        response = requests.post(self.url + "/encode/", json = payload)

        raise_if_http_error(response)

        return np.array(response.json()["embeddings"])



if __name__ == "__main__":

    # llm = LocalLLMClient("http://127.0.0.1:8000", prompt_template = "<s>[INST] {prompt} [/INST]")

    # for token in llm(
    #     "You are a smart virtual assistant that works for a banking company. Tell me the steps I need to take "
    #     "to be allowed a student loan in the state of California", 
    #     stream= True
    # ):
        
    #     print(token, end="", flush=True)
    # print()

    encoder = LocalEmbeddingsClient("http://127.0.0.1:8000")

    encoding = encoder.encode("bunda mole e seca")
    sentences = ["opa gangnam style", "arroz com feijão é gosotosão", "le fish au chocolat"]
    many_encodings = encoder.encode(sentences)


    print(encoding)
    print("-" * 10)
    print(many_encodings)

    i = np.argmax([np.dot(encoding, encoding2) for encoding2 in many_encodings])

    print(sentences[i])

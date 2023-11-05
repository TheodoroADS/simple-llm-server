from sentence_transformers import SentenceTransformer

'''
I use this file just to retrieve the encoder from huggingface hub
'''

encoder_name = r"sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(encoder_name)

model.save("all-MiniLM-L6-v2")


import torch
from langchain_community.embeddings import  HuggingFaceInstructEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel



def BKAI_emb():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HuggingFaceInstructEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder",
                                             model_kwargs={"device": device})
    return model


def miniLM_embed():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model


def BGE_embed():
    model_name = "khoa-klaytn/bge-base-en-v1.5-angle"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model

def embedding(mode):
    if mode == 'BKAI':
        return BKAI_emb()
    if mode == 'BGE':
        return BGE_embed()
    if mode == 'MiniLM':
        return miniLM_embed()
    else:
        return BGE_embed()



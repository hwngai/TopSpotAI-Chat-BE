from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
import torch
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EmbeddingModels:
    BAAI_MODEL = "BAAI/bge-small-en"
    BKAI_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    VIETNAMESE_SBERT_MODEL = "keepitreal/vietnamese-sbert"
    SUP_SIMCSE_VIETNAMESE_PHOBERT_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"


    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BAAI_emb = HuggingFaceInstructEmbeddings(model_name=self.BAAI_MODEL, model_kwargs={"device": device})
        self.BKAI_emb = HuggingFaceInstructEmbeddings(model_name=self.BKAI_MODEL, model_kwargs={"device": device})
        self.vietnamese_sbert_emb = HuggingFaceInstructEmbeddings(model_name=self.VIETNAMESE_SBERT_MODEL, model_kwargs={"device": device})
        self.sup_SimCSE_VietNamese_phobert = HuggingFaceInstructEmbeddings(model_name=self.SUP_SIMCSE_VIETNAMESE_PHOBERT_MODEL, model_kwargs={"device": device})

    def get_model(self, mode="BKAI"):
        if mode == "BAAI":
            return self.BAAI_emb
        elif mode == "BKAI":
            return self.BKAI_emb
        elif mode == "vietnamese_sbert":
            return self.vietnamese_sbert_emb
        elif mode == "sup_SimCSE_VietNamese_phobert":
            return self.sup_SimCSE_VietNamese_phobert
        elif mode == "openai":
            return OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        else:
            return None

    def list_models(self):

        models = {
            "openai": "openai",
            "BAAI": self.BAAI_MODEL,
            "BKAI": self.BKAI_MODEL,
            "vietnamese_sbert": self.VIETNAMESE_SBERT_MODEL,
            "sup_SimCSE_VietNamese_phobert": self.SUP_SIMCSE_VIETNAMESE_PHOBERT_MODEL
        }
        return models


embedding_models = EmbeddingModels()




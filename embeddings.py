from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import torch
import uuid
import os

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder",
                                                      model_kwargs={"device": device})

def generate_random_filename():
    random_filename = str(uuid.uuid4())
    return random_filename

async def text_embedding(text):
    query_result = instructor_embeddings.embed_query(text)
    return query_result

async def document_embedding(text):
    pass

def split_text(texts, chunk_size=256, chunk_overlap=25):
    name_txt = generate_random_filename() + ".txt"
    with open(name_txt, "w", encoding="utf-8") as f:
        f.write(texts)
    documents = TextLoader(name_txt).load()
    default_chunk_separators = ['\n\n', '\n', ' ', '']
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=default_chunk_separators,
    )
    chunks = text_splitter.split_documents(documents)
    os.remove(name_txt)

    return chunks


def retriever_1(query_texts, chunks, n_results):
    retriever = FAISS.from_documents(chunks, embedding = instructor_embeddings).as_retriever(
        search_kwargs={"k": n_results}
    )
    results = retriever.get_relevant_documents(query_texts)
    return results




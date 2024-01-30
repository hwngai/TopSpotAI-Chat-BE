from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import torch
import uuid
import os
import nest_asyncio
import time
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader, NewsURLLoader, SeleniumURLLoader


def generate_random_filename():
    random_filename = str(uuid.uuid4())
    return random_filename

async def text_embedding(text, instructor_embeddings):
    query_result = instructor_embeddings.embed_query(text)
    return query_result

async def document_embedding(text):
    pass

def split_text(documents, chunk_size=256, chunk_overlap=25):
    default_chunk_separators = ['\n\n', '\n', ' ', '']
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=default_chunk_separators,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def retriever(query_texts, chunks, n_results, instructor_embeddings):

    retriever = FAISS.from_documents(chunks, embedding = instructor_embeddings).as_retriever(
        search_kwargs={"k": n_results}
    )
    results = retriever.get_relevant_documents(query_texts)
    return results

def loading_webpages(website_urls, loader_class = NewsURLLoader):
    loader = loader_class(website_urls)
    docs = loader.load()
    return docs


def create_index_and_retriever(chunks, embeddings):
    index = FAISS.from_documents(chunks, embeddings)
    retriever = index.as_retriever()
    return retriever

from flashrank import *
from embeddings import embedding
from document_loaders import loading_webpages
from chunking import split_text
from langchain.vectorstores import FAISS
import chromadb
import time

instructor_embeddings = embedding(mode = 'BKAI')


def chromarank_default(query, chunks):
    ids = []
    for i in range(1, len(chunks) + 1):
        ids.append(str(i))
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="__")
    collection.add(
        documents=[doc.page_content for doc in chunks],
        ids=ids)
    results = collection.query(
        query_texts=[query],
        n_results=len(chunks))
    return results


def retriever(query_texts, urls, n_results=5):

    start_time = time.time()
    docs = loading_webpages(website_urls=urls)
    loading_time = time.time() - start_time
    print(f"loading_webpages time: {loading_time} seconds")


    start_time = time.time()
    chunks = split_text(docs)
    print(chunks, type(chunks))
    split_time = time.time() - start_time
    print(f"split_text time: {split_time} seconds")


    start_time = time.time()
    index = FAISS.from_documents(chunks, instructor_embeddings)
    faiss_time = time.time() - start_time
    print(f"FAISS index creation time: {faiss_time} seconds")

    start_time = time.time()
    retriever = index.as_retriever(search_kwargs={"k": n_results})
    results = retriever.get_relevant_documents(query_texts)
    retrieval_time = time.time() - start_time
    print(f"Retrieval time: {retrieval_time} seconds")

    start_time = time.time()
    results = chromarank_default(query_texts, chunks)
    chromarank_time = time.time() - start_time
    print(f"Chromarank time: {chromarank_time} seconds")

    return results



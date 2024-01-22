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

def loading_webpages(website_urls, loader_class = NewsURLLoader):
    nest_asyncio.apply()
    loader = loader_class(website_urls)
    # loader.requests_per_second = 1
    docs = loader.load()
    return docs


def create_index_and_retriever(chunks, embeddings):
    index = FAISS.from_documents(chunks, embeddings)
    retriever = index.as_retriever()
    return retriever



urls = ["https://tamanhhospital.vn/dau-bung/", "https://www.vinmec.com/vi/tieu-hoa-gan-mat/thong-tin-suc-khoe/vi-tri-dau-bung-canh-bao-benh-gi/", "https://hongngochospital.vn/dau-bung-canh-bao-nhieu-benh-ly-nguy-hiem/", "https://www.msdmanuals.com/vi-vn/chuy%C3%AAn-gia/r%E1%BB%91i-lo%E1%BA%A1n-ti%C3%AAu-h%C3%B3a/b%E1%BB%A5ng-c%E1%BA%A5p-v%C3%A0-ph%E1%BA%ABu-thu%E1%BA%ADt-ti%C3%AAu-h%C3%B3a/%C4%91au-b%E1%BB%A5ng-c%E1%BA%A5p-t%C3%ADnh"]
start_time = time.time()
docs = loading_webpages(website_urls = urls)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"time {urls}: {elapsed_time:.2f} giây")

print("docs: ", docs)
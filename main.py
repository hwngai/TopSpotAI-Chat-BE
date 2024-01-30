from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from Langchain.retrieval import loading_webpages,text_embedding, retriever, split_text
from LlamaIndex.index import chromarank_custom_emb
from Langchain.embedding_models import embedding_models as embedding_models_langchain
from LlamaIndex.embedding_models import embedding_models as embedding_models_llama
import time

app = FastAPI(
    title="TopSpotAI-Chat-BE"
)

def get_embedding_model(models, model_name):
    return models.get_model(model_name)

def handle_exception(e):
    raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

def measure_time(start_time, operation_name):
    try:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{operation_name}: {elapsed_time:.2f} seconds")
        return end_time
    except Exception as e:
        print(f"Error measuring time for {operation_name}: {str(e)}")
        return start_time

@app.get("/", include_in_schema=False,  tags=['docs'])
async def redirect():
    return RedirectResponse("/docs")

@app.post("/embeddings")
async def embeddings(
        request_data: dict = Body(..., example={
            "text": "Your text embedding",
            "embedding_model": "BKAI"
        })
):
    try:
        text = request_data.get("text")
        embedding_model = request_data.get("embedding_model", "BKAI")
        query_result = await text_embedding(text, get_embedding_model(embedding_models_langchain, embedding_model))
        return {"status_code": 200, "data": query_result}
    except Exception as e:
        handle_exception(e)

@app.get("/list_models_embeddings_langchain")
async def list_models_embeddings():
    try:
        models = embedding_models_langchain.list_models()
        return {"status_code": 200, "data": models}
    except Exception as e:
        handle_exception(e)

@app.get("/list_models_embeddings_llama")
async def list_models_embeddings():
    try:
        models = embedding_models_llama.list_models()
        return {"status_code": 200, "data": models}
    except Exception as e:
        handle_exception(e)

@app.post("/langchain/context")
async def query(
        request_data: dict = Body(..., example={
            "urls": ["https://tamanhhospital.vn/dau-bung/",
                     "https://www.vinmec.com/vi/tieu-hoa-gan-mat/thong-tin-suc-khoe/vi-tri-dau-bung-canh-bao-benh-gi/"],
            "query": "Nguyên nhân gây đau bụng?",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "n_results": 5,
            "embedding_model": "BKAI"
        })
):
    try:
        urls = request_data.get("urls")
        query = request_data.get("query")
        chunk_size = int(request_data.get("chunk_size", 512))
        chunk_overlap = int(request_data.get("chunk_overlap", 100))
        n_results = int(request_data.get("n_results", 5))
        embedding_model = request_data.get("embedding_model", "BKAI")

        start_time = time.time()
        docs = loading_webpages(website_urls=urls)
        end_time = measure_time(start_time, "Loading web pages")

        start_time = time.time()
        chunks = split_text(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        end_time = measure_time(start_time, "Splitting text into chunks")

        start_time = time.time()
        instructor_embeddings = get_embedding_model(embedding_models_langchain, embedding_model)
        end_time = measure_time(start_time, "Getting embedding model")

        start_time = time.time()
        results = retriever(query_texts=query, chunks=chunks, n_results=n_results,
                            instructor_embeddings=instructor_embeddings)
        end_time = measure_time(start_time, "Retrieving results")
        response = []
        for result in results:
            response.append({
                'pageContent': result.page_content,
                'metadata': {
                    'title': result.metadata['title'],
                    "annotationPosition": result.metadata['link'],
                    "description": result.metadata['description'],
                    'language': result.metadata['language']
                }
            })

        return {"status_code": 200, "data": response}
    except Exception as e:
        handle_exception(e)

@app.post("/llama_index/context")
async def query(
        request_data: dict = Body(..., example={
            "urls": ["https://tamanhhospital.vn/dau-bung/",
                     "https://www.vinmec.com/vi/tieu-hoa-gan-mat/thong-tin-suc-khoe/vi-tri-dau-bung-canh-bao-benh-gi/"],
            "query": "Nguyên nhân gây đau bụng?",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "n_results": 5,
            "embedding_model": "BAAI"
        })
):
    try:
        urls = request_data.get("urls")
        query = request_data.get("query")
        chunk_size = int(request_data.get("chunk_size", 512))
        chunk_overlap = int(request_data.get("chunk_overlap", 100))
        n_results = int(request_data.get("n_results", 5))

        embedding_model = request_data.get("embedding_model", "BAAI")
        embed_model = get_embedding_model(embedding_models_llama, embedding_model)
        context = await chromarank_custom_emb(query, urls, embed_model=embed_model, chunk_size=chunk_size, chunk_ovelap=chunk_overlap, n_results=n_results)
        return {"status_code": 200, "data": context}
    except Exception as e:
        handle_exception(e)

if __name__ == "__main__":
    pass
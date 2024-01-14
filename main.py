from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, RedirectResponse
from embeddings import text_embedding, retriever_1, split_text
from prompts import create_prompt


app = FastAPI(
    title="TopSpotAI-Chat-BE"
)

@app.get("/", include_in_schema=False,  tags=['docs'])
async def redirect():
    return RedirectResponse("/docs")


@app.post("/embeddings")
async def embeddings(
        request_data: dict = Body(..., example={
            "text": "Your text embedding"
        })
):
    try:
        text = request_data.get("text")
        query_result = await text_embedding(text)
        return {"status_code": 200, "data": query_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query")
async def query(
        request_data: dict = Body(..., example={
            "promt": "Your prompt text",
            "documnets": "Your documents text",
            "chunk_size": 256,
            "chunk_overlap": 25,
            "n_results": 5,
        })
):
    try:
        prompt = request_data.get("promt")
        documnets = request_data.get("documnets")
        chunk_size = int(request_data.get("chunk_size", 256))
        chunk_overlap = int(request_data.get("chunk_overlap", 25))
        n_results = int(request_data.get("n_results", 5))

        chunks = split_text(documnets, chunk_size, chunk_overlap)
        results = retriever_1(query_texts = prompt, chunks = chunks, n_results = n_results)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        prompt_result = create_prompt(context = context_text, question = prompt)

        return {"status_code": 200, "data": prompt_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}"
)

if __name__ == "__main__":
    pass
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, RedirectResponse
from ranking import retriever
from prompts import create_prompt
import ast


app = FastAPI(
    title="TopSpotAI-Chat-BE"
)

@app.get("/", include_in_schema=False,  tags=['docs'])
async def redirect():
    return RedirectResponse("/docs")


@app.post("/query")
async def query(
        request_data: dict = Body(..., example={
            "query": "Your query text",
            "urls": "Your urls",
            "chunk_size": 256,
            "chunk_overlap": 25,
            "n_results": 5,
        })
):
    try:
        query_text = request_data.get("query")
        urls = ast.literal_eval(request_data.get("urls"))

        chunk_size = int(request_data.get("chunk_size", 256))
        chunk_overlap = int(request_data.get("chunk_overlap", 25))
        n_results = int(request_data.get("n_results", 5))

        results = retriever(query_text, urls, n_results)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        prompt_result = create_prompt(context = context_text, question = query_text)

        return {"status_code": 200, "data": prompt_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}"
)

if __name__ == "__main__":
    pass
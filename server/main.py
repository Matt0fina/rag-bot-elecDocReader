import uvicorn
import json
from pydantic import BaseModel

from fastapi import FastAPI

from api.routes import router
from core.vector_database import initialize_empty_vectorstores, load_vectorstore
from core.llm_chain_factory import build_llm_chain
from utils.logger import logger

class ChatRequest(BaseModel):
  query: str
  workspace: str

app = FastAPI(title="RAG PDFBot", description="Chat with multiple PDFs :books:")
app.include_router(router)

@app.post("/extract")
async def chat_endpoint(request: ChatRequest):
    try:
        # Hardcoding the provider and model for the pitch
        model_provider = "groq"
        model_name = "llama-3.1-8b-instant"
        
        # Build chain using the functions from core folders
        vector_store = load_vectorstore(model_provider)
        chain = build_llm_chain(model_provider, model_name, vector_store)
        
        # Run the query
        result = chain.invoke({"input": request.query}) 
        
        # Extract the output string safely
        if isinstance(result, dict):
            raw_output = result.get("answer") or result.get("result", "")
        else:
            raw_output = result 
        
        # Clean and parse the JSON output from the LLM
        clean_json_string = raw_output.strip()

        if clean_json_string.startswith("```json"):
            clean_json_string = clean_json_string[7:]
        if clean_json_string.startswith("```"):
            clean_json_string = clean_json_string[3:]
        if clean_json_string.endswith("```"):
            clean_json_string = clean_json_string[:-3]
            
        clean_json_string = clean_json_string.strip()

        try:
            parsed_data = json.loads(clean_json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Raw output was: {raw_output}")
            return {"answer": raw_output, "parameters": []}
            
    except Exception as e:
        logger.error(f"Backend Crash: {str(e)}")
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
  logger.info("Starting up app...")
  initialize_empty_vectorstores()
  logger.info("Startup complete.")

if __name__ == "__main__":
  logger.info("Running app...")
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)     

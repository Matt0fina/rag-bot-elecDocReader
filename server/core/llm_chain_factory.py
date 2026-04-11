from config.settings import GROQ_API_KEY, GOOGLE_API_KEY

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.logger import logger

def get_prompt():
    logger.debug("Creating structured JSON chat prompt template.")
    # Note: We use double curly braces {{ }} to escape standard JSON syntax in LangChain prompts
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Test Engineer assistant.
        Extract precise component parameters from the context.
        
        CRITICAL: You MUST return your entire response as a valid, raw JSON object. Do not include markdown formatting like ```json.
        
        Format your response EXACTLY like this:
        {{
            "answer": "Your detailed textual explanation and analysis here...",
            "parameters": [
                {{"Parameter": "V_CMR (Min)", "Value": "V_SS - 0.3V", "Condition": "V_DD = 5V"}},
                {{"Parameter": "Max Supply Voltage", "Value": "7.0V", "Condition": "Absolute Maximum"}}
            ]
        }}
        
        - If no specific parameters are found, return an empty list [] for "parameters".
        - Do not hallucinate values."""),
        ("human", "Datasheet Context:\n{context}\n\nEngineer's Query:\n{input}")
    ])

# def get_prompt():
#   logger.debug("Creating hardware specific chat prompt template.")
#   return ChatPromptTemplate.from_messages([
#     ("system", """You are an expert Test Engineer specializing in automated component characterization.
#      Extract precise component parameters (e.g., R_DS(on), transconductance, absolute maximum ratings) from the datasheet context.
     
#      - If the user asks for simulation parameters, format the output as a valid .model directive suitable for LTSpice.
#      - If the user asks for layout details, prioritize pad dimensions and thermal resistance (RthJC).
#      - Do not hallucinate. If a value is missing, state 'Parameter not found in datasheet.'"""),
#     ("human", "Context:\n{context}\n\n\nEngineer's Query:\n{input}")
#   ])

def get_llm(model_provider: str, model: str):
  logger.debug(f"Initializing LLM for {model_provider} - {model}")
  if model_provider == "groq":
    return ChatGroq(model=model, api_key=GROQ_API_KEY)
  elif model_provider == "gemini":
    return ChatGoogleGenerativeAI(model=model, api_key=GOOGLE_API_KEY)
  else:
    logger.error(f"Unsupported LLM Provider: {model_provider}")
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

def build_llm_chain(model_provider: str, model: str, vectorstore):
  logger.debug(f"Building LLM chain for provider: {model_provider}, model: {model}")
  prompt = get_prompt()
  llm = get_llm(model_provider, model)
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

  return create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt=prompt)
  )

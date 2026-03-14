from config.settings import GROQ_API_KEY, GOOGLE_API_KEY

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.logger import logger


def get_prompt():
  logger.debug("Creating hardware specific chat prompt template.")
  return ChatPromptTemplate.from_messages([
    ("system", """You are an expert Test Engineer specializing in automated component characterization.
     Extract precise component parameters (e.g., R_DS(on), transconductance, absolute maximum ratings) from the datasheet context.
     
     - If the user asks for simulation parameters, format the output as a valid .model directive suitable for LTSpice.
     - If the user asks for layout details, prioritize pad dimensions and thermal resistance (RthJC).
     - Do not hallucinate. If a value is missing, state 'Parameter not found in datasheet.'"""),
    ("human", "Context:\n{context}\n\n\nEngineer's Query:\n{input}")
  ])

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

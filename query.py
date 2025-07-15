import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# --- Load keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Load existing Pinecone index and embeddings ---
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

index_name = "cmu-courses"
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)

# --- Gemini model ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# --- Prompt and RAG chain setup ---
system_prompt = (
    "You are a helpful assistant answering questions about Carnegie Mellon University Computer Science courses. "
    "Use only the retrieved course descriptions as context. "
    "If the answer is not found in the context, say 'I don't know.' "
    "Include course numbers and names in your answers when available.\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=qa_chain)

# --- Ask user query ---
while True:
    query = input("\nAsk a question about CMU CS courses (or type 'exit'): ")
    if query.lower() in ("exit", "quit"):
        break
    response = rag_chain.invoke({"input": query})
    print("\n--- Answer ---")
    print(response["answer"])

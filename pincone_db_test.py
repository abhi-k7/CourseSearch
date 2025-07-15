import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# --- Load Keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Step 1: Load your course documents ---
loader = DirectoryLoader(path="./documents", glob="*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

# --- Step 2: Split them using RecursiveCharacterTextSplitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = splitter.split_documents(raw_docs)
print(f"Loaded and split {len(documents)} documents.")

# --- Step 3: Create sentence-transformer embeddings ---
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# --- Step 4: Initialize Pinecone and Create Index if Needed ---
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "cmu-courses"
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # dimension of BGE
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- Step 5: Create the vector store ---
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding,
    index_name=index_name,
)

# --- Step 6: Setup Gemini LLM via LangChain ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
)

# --- Step 7: RAG Prompt + Chain ---
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

# --- Step 8: Ask a question ---
query = input("\nAsk a question about CMU CS courses: ")
response = rag_chain.invoke({"input": query})
print("\n--- Answer ---")
print(response["answer"])
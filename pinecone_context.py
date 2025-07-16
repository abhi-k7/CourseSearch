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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


# --- Load keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

loader = DirectoryLoader(path="./documents", glob="*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

course_map = {}
for doc in raw_docs:
    filename = os.path.basename(doc.metadata["source"])
    course_map[filename] = doc.page_content

# --- Step 2: Split them using RecursiveCharacterTextSplitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = []

for doc in raw_docs:
    filename = os.path.basename(doc.metadata["source"])
    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata["course_id"] = filename
        documents.append(chunk)

print(f"Loaded and split {len(documents)} documents.")

# --- Load existing Pinecone index and embeddings ---
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

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

dense_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 20

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever], weights=[0.7, 0.3]
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

rag_chain = create_retrieval_chain(retriever=ensemble_retriever, combine_docs_chain=qa_chain)
print("Using ensemble retriever")

# --- Ask user query ---
while True:
    query = input("\nAsk a question about CMU CS courses (or type 'exit'): ")
    if query.lower() in ("exit", "quit"):
        break

    docs = ensemble_retriever.get_relevant_documents(query)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: -x[0])]
    top_docs = ranked_docs[:8]

    top_course_ids = list({doc.metadata["course_id"] for doc in top_docs})

    full_context = "\n\n".join([course_map[course_id] for course_id in top_course_ids])

    response = qa_chain.invoke({
        "input": query,
        "context": full_context
    })

    print("\n--- Answer ---")
    print(response)

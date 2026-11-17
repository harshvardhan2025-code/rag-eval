import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

import warnings
warnings.filterwarnings("ignore")


# Models (the same ones as in the chatbot)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Indexing (same as the chatbot, but it reads a file path instead of an upload)

def load_pdf(pdf_path):
    """Read the PDF and return one Document per page."""
    reader = PdfReader(pdf_path)

    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            Document(
                page_content=text,
                metadata={"source": os.path.basename(pdf_path), "page": page_number},
            )
        )
    return pages


def split_pages(pages):
    """Cut the pages into small overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(pages)


def create_vector_store(chunks):
    """Embed every chunk and store the vectors in Chroma."""
    vector_store = Chroma(
        collection_name=f"pdf_{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    vector_store.add_documents(chunks)
    return vector_store


# Answering (same prompt and model call as the chatbot)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a PDF document.

Rules:
1. Use ONLY the context below. Do not use outside knowledge.
2. After every fact, add the number of the source it came from, like [1] or [2].
3. If the answer is not in the context, say: "I could not find this in the document."

Context:
{context}"""


def format_context(docs):
    """Turn the retrieved chunks into one numbered block of text for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata["page"] + 1
        parts.append(f"[{i}] (page {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def ask_model(question, docs, history):
    """Send the context, the chat history and the question to the model."""
    context = format_context(docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=question))

    response = model.invoke(messages)
    return response.content[0]['text']


# New for evaluation: one function that runs the whole pipeline for one question

def build_vector_store(pdf_path):
    pages = load_pdf(pdf_path)
    chunks = split_pages(pages)
    print(f"Indexed {len(pages)} pages as {len(chunks)} chunks")
    return create_vector_store(chunks)


def answer_question(vector_store, question):
    """Return the answer AND the chunks that were retrieved (the retrieval context)."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    answer = ask_model(question, docs, [])
    retrieval_context = [doc.page_content for doc in docs]   # in ranked order

    return answer, retrieval_context

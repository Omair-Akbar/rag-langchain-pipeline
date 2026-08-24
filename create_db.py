import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

try:
    load_dotenv()

    print("API key loaded:", bool(os.getenv("MISTRAL_API_KEY")))

    doc = PyPDFLoader("documentloader/deep_learning_10_pages.pdf")
    text = doc.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=10
    )

    chunks = splitter.split_documents(text)

    print("Chunks:", len(chunks))


    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} ---")
        print(repr(chunk.page_content))

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma-db"
    )
    print("-done-")
except Exception as e:
    print(f"An error occurred: {e}")
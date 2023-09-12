import os

import pinecone
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

if os.path.exists("env.py"):
    import env

pinecone.init(api_key=os.environ.get("PINECONE_SECRET_KEY"),
              environment=os.environ.get("PINECONE_ENVIRONMENT_REGION")
              )


def ingest_text():
    loader = TextLoader("medium_blog.txt", encoding="utf-8")
    raw_text = loader.load()
    print("Loaded text")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_text)
    print("Split documents into {len(documents)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name="medium-text-db")
    print("Successfully loaded vectors into Pinecone. ")

if __name__ == "__main__":
    ingest_text()

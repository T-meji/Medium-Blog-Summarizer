import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

if os.path.exists("env.py"):
    import env

pinecone.init(api_key=os.environ.get("PINECONE_SECRET_KEY"),
              environment=os.environ.get("PINECONE_ENVIRONMENT_REGION")
              )
def run_llm(query: str):
    embeddings = OpenAIEmbeddings

    doc_search = Pinecone.from_existing_index(
        index_name="medium-txt-db",
        embedding=embeddings
    )

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1, verbose=True)

    qa = RetrievalQA.from_chain_type(chain_type="stuff", llm=chat_model, retriever=doc_search.as_retriever())

    return qa({"query":query})
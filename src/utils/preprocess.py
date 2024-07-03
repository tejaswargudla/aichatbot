
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import dotenv_values
import os
import sys

path = os.getcwd()
if "/aichatbot" in path:
    path = path.rsplit("/aichatbot", 1)[0]
    path = path + "/aichatbot"
    if path not in sys.path:
        sys.path.insert(0, path)
    cfg = dotenv_values(f'{path}/.env')
else:
    cfg = dotenv_values(".env")

def get_mongo_client():
    """Mongo db client object.
    Parameters: None
    Returns
    -------
    mongo_client : MongoClient
       mongo db client.
        
   """
    MONGODB_ATLAS_URI = f"mongodb+srv://{cfg["username"]}:{cfg["password"]}@cluster0.y9ct062.mongodb.net/?appName=Cluster0"
    mongo_client = MongoClient(MONGODB_ATLAS_URI)
    return mongo_client

def get_vectorstore(db_name, collection_name, index_name, embeddings = OpenAIEmbeddings()):
    """Mongo db vectorstore object.
    Parameters
    ----------
    db_name : str
        mongo db database name.
    collection_name: str
        mongo db collection name.
    index_name: str
        Name os the search index name.
    embeddings: Embeddings, default=OpenAIEmbeddings()

    Returns
    -------
    mongo_client : MongoClient
       mongo db client.
   """
    mongo_client = get_mongo_client()
    collection = mongo_client[db_name][collection_name]
    vectorstore = MongoDBAtlasVectorSearch(collection, embeddings,index_name=index_name)
    return vectorstore

def preprocessor():
    """Processing the policy document and loading into mongodb.
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        
   """
    vectorstore = get_vectorstore("aichatbot","policy","policy_search_index")
    with open("src/data/hr_policy.txt", "r") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=50,
                                                   seperators=["\n", "\n\n", " "]
                                                  )
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        vectorstore.add_texts(
            [chunk],
            metadatas=[{"source": "text_file", "chunk_id": i}],
            ids=[f"chunk_{i}"]
        )
    
    
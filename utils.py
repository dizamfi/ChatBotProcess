import os
import tempfile
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

FILE_LIST = "archivos.txt"
PINECONE_API_KEY = "461faa64-b7ca-4a71-bfe9-bf6f472b905d"
INDEX_NAME = 'chatgps'

# Inicializa Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Verifica si el índice existe y créalo si es necesario
if INDEX_NAME not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Reemplaza con las dimensiones de tu modelo
        metric="cosine",  # Reemplaza con la métrica de tu modelo
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Conéctate al índice
index = pc.Index(INDEX_NAME)

def load_name_files(path):
    archivos = []
    with open(path, "r") as file:
        for line in file:
            archivos.append(line.strip())
    return archivos

def save_name_files(path, new_files):
    old_files = load_name_files(path)
    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    return old_files

def clean_files(path):
    with open(path, "w") as file:
        pass
    index.delete(delete_all=True)
    return True

def text_to_pinecone(pdf):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    text = loader.load()

    with st.spinner(f'Creando embedding del fichero: {pdf.name}'):
        create_embeddings(pdf.name, text)

    temp_dir.cleanup()

    return True

def create_embeddings(file_name, text):
    print(f"Creando embeddings del archivo: {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_documents([chunk.page_content])[0]
        vectors.append({
            'id': f"{file_name}_{i}",
            'values': embedding,
            'metadata': {'text': chunk.page_content, 'source': file_name, 'chunk': i}  # Agregar más metadatos si es necesario
        })

    # Subir los vectores al índice de Pinecone
    index.upsert(vectors)
        
    return True



# Aquí es donde se utilizan las funciones
# archivos = load_name_files(FILE_LIST)







# import os
# import streamlit as st

# import pinecone
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Pinecone
# from langchain.embeddings import HuggingFaceEmbeddings 


# FILE_LIST = "archivos.txt"
# PINECONE_API_KEY = "461faa64-b7ca-4a71-bfe9-bf6f472b905d"
# PINECONE_ENV = "Añadir Pinecone Env"
# INDEX_NAME = 'taller'

# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENV
# )


# def save_name_files(path, new_files):

#     old_files = load_name_files(path)

#     with open(path, "a") as file:
#         for item in new_files:
#             if item not in old_files:
#                 file.write(item + "\n")
#                 old_files.append(item)
    
#     return old_files


# def load_name_files(path):

#     archivos = []
#     with open(path, "r") as file:
#         for line in file:
#             archivos.append(line.strip())

#     return archivos


# def clean_files(path):
#     with open(path, "w") as file:
#         pass
#     index = pinecone.Index(INDEX_NAME)
#     index.delete(delete_all=True)

#     return True


# def text_to_pinecone(pdf):

#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, pdf.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(pdf.getvalue())

#     loader = PyPDFLoader(temp_filepath)
#     text = loader.load()

#     with st.spinner(f'Creando embedding fichero: {pdf.name}'):
#         create_embeddings(pdf.name, text)

#     return True


# def create_embeddings(file_name, text):
#     print(f"Creando embeddings del archivo: {file_name}")

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100,
#         length_function=len
#         )        
    
#     chunks = text_splitter.split_documents(text)

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         )
    
#     Pinecone.from_documents(
#         chunks,
#         embeddings,
#         index_name=INDEX_NAME)
        
#     return True
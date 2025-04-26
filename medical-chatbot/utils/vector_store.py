import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

def get_vector_store():
    """
    Configura y devuelve la base de datos vectorial con información médica.
    """
    # Directorio donde se almacenarán los datos de Chroma
    persist_directory = "chroma_db"
    
    # Verificar si la base de datos ya existe
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        # Cargar base de datos existente
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return db
    else:
        # Crear nueva base de datos
        # 1. Cargar documentos
        loader = DirectoryLoader(
            "./data/medical_knowledge/",
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # 2. Dividir documentos en chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3. Crear embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # 4. Crear y persistir base de datos vectorial
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        db.persist()
        return db
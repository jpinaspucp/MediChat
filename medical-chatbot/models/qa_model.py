import os
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
import dotenv

# Intentar cargar variables de entorno
dotenv.load_dotenv()

def get_medical_qa_model():
    """
    Devuelve el modelo de lenguaje para preguntas y respuestas médicas.
    """
    # Verificar si hay un token de API de Hugging Face
    huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    if huggingface_token:
        # Si hay token, usar Hugging Face
        # Utilizamos un modelo compatible con las tareas soportadas por LangChain
        
        # Opción 1: Modelo T5 multilingüe (soporta español)
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Modelo multilingüe compatible con text2text-generation
            model_kwargs={"temperature": 0.7, "max_length": 512},
            huggingfacehub_api_token=huggingface_token
        )
        
        # Opción 2: Modelo GPT en español (descomentar para usar)
        # return HuggingFaceHub(
        #     repo_id="datificate/gpt2-small-spanish",  # Modelo GPT en español
        #     model_kwargs={"temperature": 0.7, "max_length": 512},
        #     huggingfacehub_api_token=huggingface_token
        # )
        
        # Opción 3: Modelo multilingüe BLOOM (descomentar para usar)
        # return HuggingFaceHub(
        #     repo_id="bigscience/bloom-560m",  # Modelo multilingüe más pequeño
        #     model_kwargs={"temperature": 0.7, "max_length": 512},
        #     huggingfacehub_api_token=huggingface_token
        # )
    else:
        # Mensaje para el usuario
        print("AVISO: No se encontró el token de API de Hugging Face.")
        print("Por favor, obtenga un token en https://huggingface.co/settings/tokens")
        print("Luego, cree un archivo .env en el directorio principal con la línea:")
        print("HUGGINGFACEHUB_API_TOKEN=su_token_aqui")
        
        # Alternativa: usar un modelo local o un enfoque alternativo
        # Por ejemplo, podríamos usar una instancia simulada para desarrollo
        from langchain.llms.fake import FakeListLLM
        return FakeListLLM(
            responses=["Lo siento, no puedo procesar esta consulta sin un modelo de lenguaje configurado correctamente."]
        )
# Chatbot Médico en Español

Este proyecto implementa un chatbot médico en español utilizando LangChain y Chainlit. El chatbot puede mantener conversaciones sobre síntomas, sugerir posibles condiciones médicas y recomendar especialistas.

## Requisitos

- Python 3.10 o superior
- Ambiente conda o virtual environment

## Configuración

1. Clonar el repositorio
2. Instalar dependencias:
```
pip install -r requirements.txt
```

3. **Importante:** Configurar el token de API de Hugging Face:

   a. Registrarse en [Hugging Face](https://huggingface.co/) y generar un token en [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   
   b. Crear un archivo `.env` en el directorio raíz del proyecto con el siguiente contenido:
   ```
   HUGGINGFACEHUB_API_TOKEN=tu_token_aqui
   ```
   
   c. Reemplazar `tu_token_aqui` con tu token real de Hugging Face

## Modelos Compatibles

El proyecto está configurado para usar el modelo `google/flan-t5-base` por defecto, que es un modelo multilingüe compatible con LangChain. También se incluyen configuraciones adicionales (comentadas) para:

- `datificate/gpt2-small-spanish`: Un modelo GPT-2 entrenado específicamente para español
- `bigscience/bloom-560m`: Una versión más pequeña del modelo BLOOM multilingüe

Para cambiar el modelo, edita el archivo `models/qa_model.py` y descomenta la opción deseada.

**Nota:** LangChain solo soporta modelos con las siguientes tareas: 'translation', 'summarization', 'conversational', 'text-generation', 'text2text-generation'.

## Ejecución

Para iniciar el chatbot:

```
chainlit run app.py
```

## Estructura del Proyecto

- `app.py`: Aplicación principal Chainlit
- `agent/`: Módulos para la lógica del agente conversacional
- `models/`: Configuración de modelos de lenguaje
- `data/`: Datos de especialistas y conocimiento médico
- `utils/`: Utilidades para la base de datos vectorial y otras funciones

## Personalización

- Para agregar nuevos síntomas y condiciones, editar los mapeos en `agent/recommender.py`
- Para modificar el comportamiento conversacional, editar `agent/conversation.py`
- Para cambiar el modelo de lenguaje, editar `models/qa_model.py` 
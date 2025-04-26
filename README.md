# MediChat 🏥 - Chatbot Médico en Español

![MediChat Banner](https://i.ibb.co/BfvMjkc/medichat-banner.png)

MediChat es un chatbot médico en español que utiliza inteligencia artificial para ayudar a identificar posibles condiciones médicas basadas en síntomas, proporcionar información sobre enfermedades comunes y recomendar especialistas médicos apropiados.

## 🌟 Características

- 🔍 **Análisis de síntomas**: Identifica síntomas a partir de descripciones en lenguaje natural
- 🧠 **Sugerencia de condiciones médicas**: Propone posibles diagnósticos basados en los síntomas reportados
- 👨‍⚕️ **Recomendación de especialistas**: Sugiere qué tipo de médico consultar según las condiciones identificadas
- 📚 **Base de conocimiento médico**: Información sobre enfermedades comunes, síntomas y tratamientos
- 💬 **Interfaz conversacional amigable**: Experiencia de chat natural mediante Chainlit
- 🌐 **Soporte multilingüe**: Optimizado para español con capacidad de entender términos médicos

## 🛠️ Tecnologías

- **LangChain**: Framework para crear aplicaciones impulsadas por LLMs
- **Chainlit**: Interfaz de chat para aplicaciones LLM
- **RAG (Retrieval-Augmented Generation)**: Para proporcionar respuestas precisas basadas en conocimiento médico
- **Chroma DB**: Base de datos vectorial para búsqueda semántica
- **Sentence Transformers**: Modelos de embedding multilingües
- **Hugging Face**: Modelos de lenguaje para procesamiento y generación de texto

## 🚀 Instalación

### Prerrequisitos

- Python 3.10 o superior
- Ambiente conda o venv recomendado

### Pasos de instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TuUsuario/MediChat.git
   cd MediChat
   ```

2. Crear y activar un entorno virtual:
   ```bash
   conda create -n medichat python=3.10
   conda activate medichat
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurar token de API (necesario para los modelos de Hugging Face):
   Crear archivo `.env` en el directorio raíz:
   ```
   HUGGINGFACEHUB_API_TOKEN=tu_token_aqui
   ```
   Puedes obtener tu token en [Hugging Face](https://huggingface.co/settings/tokens)

## 🏃‍♂️ Ejecución

Para iniciar el chatbot:

```bash
chainlit run app.py
```

Luego, abre tu navegador en `http://localhost:8000`

## 🏗️ Arquitectura

![Arquitectura](https://i.ibb.co/41sQpBt/medichat-architecture.png)

MediChat utiliza una arquitectura modular con los siguientes componentes:

1. **Interfaz de usuario (Chainlit)**: Proporciona una interfaz web interactiva para la conversación.

2. **Agente de Conversación**: Gestiona el flujo del diálogo a través de diferentes etapas (saludo, recopilación de síntomas, recomendaciones).

3. **Sistema de QA Médico**: Extrae síntomas, identifica posibles condiciones y responde preguntas médicas generales.

4. **Recomendador de Especialistas**: Proporciona sugerencias de especialistas médicos basados en las condiciones identificadas.

5. **Base de Datos Vectorial**: Almacena conocimiento médico para búsquedas semánticas utilizando Chroma DB.

## 📚 Base de Conocimiento

La base de conocimiento incluye información sobre:
- Enfermedades comunes y sus síntomas
- Relaciones entre síntomas y posibles condiciones
- Especialidades médicas apropiadas para diferentes condiciones

## 🔄 Flujo de Conversación

1. **Saludo**: El chatbot se presenta y pregunta por los síntomas.
2. **Recopilación de síntomas**: El usuario describe sus síntomas y el chatbot hace preguntas de seguimiento.
3. **Análisis**: El sistema analiza los síntomas y busca posibles condiciones médicas.
4. **Recomendación**: El chatbot sugiere posibles condiciones y especialistas apropiados.
5. **Preguntas adicionales**: El usuario puede hacer preguntas sobre las condiciones o solicitar más información.

## ⚠️ Descargo de responsabilidad médica

MediChat es una herramienta informativa y no sustituye la consulta con profesionales médicos. La información proporcionada no constituye diagnóstico médico, tratamiento o consejo. Siempre consulte a un profesional de la salud calificado para problemas médicos.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 📫 Contacto

Tu Nombre - [tu-email@example.com](mailto:tu-email@example.com)

Enlace del proyecto: [https://github.com/TuUsuario/MediChat](https://github.com/TuUsuario/MediChat)

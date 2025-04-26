import chainlit as cl
from agent.conversation import MedicalConversationAgent
import asyncio
import random

@cl.on_chat_start
async def start():
    agent = MedicalConversationAgent()
    cl.user_session.set("agent", agent)
    
    # Usar streaming nativo de Chainlit con cl.Message y cl.Step
    message = cl.Message(content="")
    
    welcome_message = "¡Hola! Soy un asistente médico virtual. Puedo ayudarte a identificar posibles condiciones basadas en tus síntomas. Recuerda que no soy un médico y mis sugerencias no reemplazan un diagnóstico profesional. ¿En qué puedo ayudarte hoy?"
    
    for i in range(0, len(welcome_message), 10):
        chunk = welcome_message[i:i+10]
        if i == 0:
            # Primera parte, enviar el mensaje
            await message.stream_token(chunk)
        else:
            # Partes subsiguientes, actualizar el mensaje
            await message.stream_token(chunk)
        
        # Pequeña pausa aleatoria
        await asyncio.sleep(0.05 + random.random() * 0.05)
    
    # Finalizar el mensaje
    await message.send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    
    # Obtener la respuesta completa del agente
    full_response = await agent.process_message(message.content)
    
    # Usar streaming nativo de Chainlit
    response_message = cl.Message(content="")
    
    # Dividir la respuesta en fragmentos para streaming
    for i in range(0, len(full_response), 10):
        chunk = full_response[i:i+10]
        await response_message.stream_token(chunk)
        
        # Pausa aleatoria para efecto de escritura
        if "." in chunk or "!" in chunk or "?" in chunk or "\n" in chunk:
            await asyncio.sleep(0.1 + random.random() * 0.1)  # Pausa más larga para puntuación
        else:
            await asyncio.sleep(0.03 + random.random() * 0.05)  # Pausa normal
    
    # Finalizar el mensaje
    await response_message.send()
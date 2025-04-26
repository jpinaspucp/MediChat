from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from agent.medical_qa import MedicalQASystem
from agent.recommender import SpecialistRecommender
import re

class MedicalConversationAgent:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.qa_system = MedicalQASystem()
        self.recommender = SpecialistRecommender()
        self.conversation_stage = "greeting"
        self.previous_recommendation_request = False
        self.collected_symptoms = []  # Lista de síntomas acumulados durante la conversación
        self.symptom_confidence = {}  # Diccionario para rastrear confianza en cada síntoma reportado
        
    async def process_message(self, message: str):
        # Actualizar memoria
        self.memory.chat_memory.add_user_message(message)
        
        # Verificar si el mensaje es una despedida
        if self._is_farewell(message):
            response = self._handle_farewell()
        # Determinar etapa de conversación
        elif self.conversation_stage == "greeting":
            response = self._handle_greeting(message)
        elif self.conversation_stage == "symptom_collection":
            response = await self._handle_symptoms(message)
        elif self.conversation_stage == "recommendation":
            response = self._handle_recommendation(message)
        else:
            # Manejar conversación general
            response = await self.qa_system.answer_medical_question(message, self.memory)
        
        # Actualizar memoria con la respuesta
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _is_farewell(self, message: str):
        """Detecta si el mensaje es una despedida o agradecimiento final"""
        farewell_patterns = [
            r"(?i)gracias",
            r"(?i)adiós",
            r"(?i)hasta luego",
            r"(?i)chao",
            r"(?i)nos vemos",
            r"(?i)eso es todo",
            r"(?i)eso sería todo",
            r"(?i)te agradezco",
            r"(?i)muchas gracias",
            r"(?i)ok\s*gracias"
        ]
        
        # También detectamos respuestas cortas negativas después de preguntas
        if re.match(r"(?i)^(no|nada|ninguno)\.?$", message.strip()):
            last_ai_messages = [msg.content for msg in self.memory.chat_memory.messages if hasattr(msg, 'type') and msg.type == 'ai']
            if last_ai_messages and any(s in last_ai_messages[-1] for s in ["¿quieres", "¿Quieres", "¿deseas", "¿necesitas", "¿puedo", "¿te gustaría"]):
                return True
        
        return any(re.search(pattern, message) for pattern in farewell_patterns)
    
    def _handle_farewell(self):
        """Maneja las despedidas del usuario"""
        self.conversation_stage = "greeting"  # Reiniciar para próxima conversación
        self.recommender.reset_recommendations()  # Limpiar recomendaciones previas
        
        return "Ha sido un placer ayudarte. Recuerda que siempre es importante consultar con un profesional médico para un diagnóstico adecuado. ¡Cuídate y hasta pronto!"
    
    def _handle_greeting(self, message: str):
        self.conversation_stage = "symptom_collection"
        return "Para ayudarte mejor, necesito saber tus síntomas. ¿Podrías describir cómo te sientes? Por ejemplo, ¿tienes dolor, fiebre, u otros síntomas?"
    
    async def _handle_symptoms(self, message: str):
        # Tratar "no" como posible terminación de la conversación si no hay contexto previo
        if re.match(r"(?i)^no\.?$", message.strip()) and not self.qa_system.last_conditions:
            return "Entiendo. Si en algún momento necesitas asistencia con síntomas o tienes preguntas médicas, estoy aquí para ayudarte."
            
        # Extraer síntomas del mensaje
        symptoms = await self.qa_system.extract_symptoms(message)
        
        # Verificación explícita para detectar texto del prompt en los síntomas
        # y filtrarlo para evitar respuestas inconsistentes
        cleaned_symptoms = []
        for symptom in symptoms:
            if ("list" in symptom.lower() or "lista" in symptom.lower() or 
                "separated" in symptom.lower() or ":" in symptom):
                continue
            cleaned_symptoms.append(symptom)
        
        # Si no se detectaron síntomas válidos pero el mensaje parece contener información médica
        # intentar extraer manualmente algunos síntomas comunes
        if not cleaned_symptoms and len(message) > 10:
            common_symptoms = ["dolor", "fiebre", "tos", "mareo", "náusea", "fatiga", 
                               "cansancio", "picazón", "vómito", "diarrea"]
            for symptom in common_symptoms:
                if symptom in message.lower():
                    # Intentar extraer el contexto alrededor del síntoma
                    match = re.search(f"\\b{symptom}\\s+\\w+\\s*\\w*", message.lower())
                    if match:
                        cleaned_symptoms.append(match.group(0))
                    else:
                        cleaned_symptoms.append(symptom)
        
        if len(cleaned_symptoms) > 0:
            # Si tenemos síntomas suficientes, pasar a recomendaciones
            conditions = await self.qa_system.get_possible_conditions(cleaned_symptoms)
            self.conversation_stage = "recommendation"
            
            # Verificar que las condiciones sean válidas
            if not conditions or any("síntoma" in c.lower() or "symptom" in c.lower() for c in conditions):
                conditions = self._get_fallback_conditions(cleaned_symptoms)
                self.qa_system.last_conditions = conditions
            
            # Mejorar formato de respuesta
            response = "Basado en tus síntomas, podría tratarse de: \n"
            for condition in conditions:
                response += f"- {condition}\n"
            response += "\nRecuerda que esto no es un diagnóstico médico. ¿Te gustaría que te recomiende especialistas para consultar?"
            return response
        else:
            # Seguir recopilando síntomas
            return "No he podido identificar claramente tus síntomas. ¿Podrías describir con más detalle cómo te sientes? Por ejemplo, si tienes dolor, fiebre, mareo u otros síntomas específicos."
    
    def _get_fallback_conditions(self, symptoms):
        """Proporciona condiciones de respaldo basadas en síntomas comunes"""
        symptom_conditions = {
            "dolor de cabeza": ["Migraña", "Cefalea tensional", "Sinusitis"],
            "dolor": ["Inflamación muscular", "Artritis", "Fibromialgia"],
            "pecho": ["Angina de pecho", "Bronquitis", "Ansiedad"],
            "fiebre": ["Gripe", "Infección viral", "COVID-19"],
            "tos": ["Resfriado común", "Bronquitis", "Asma"],
            "mareo": ["Vértigo", "Hipotensión", "Anemia"],
            "náusea": ["Gastroenteritis", "Migraña", "Intoxicación alimentaria"],
            "cansancio": ["Anemia", "Hipotiroidismo", "Depresión"],
            "abdominal": ["Gastritis", "Síndrome de intestino irritable", "Indigestión"]
        }
        
        matched_conditions = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, conditions in symptom_conditions.items():
                if key in symptom_lower:
                    matched_conditions.extend(conditions)
        
        # Eliminar duplicados y limitar a 3
        unique_conditions = list(dict.fromkeys(matched_conditions))
        if unique_conditions:
            return unique_conditions[:3]
        else:
            return ["Posible afección temporal", "Estrés", "Condición leve"]
    
    def _handle_recommendation(self, message: str):
        # Primero verificar si el usuario quiere terminar
        if re.search(r"(?i)^no\.?$|^no,? gracias", message):
            return "Entendido. Espero haberte sido de ayuda con la información proporcionada. Si necesitas algo más en el futuro, no dudes en consultarme."
            
        # Detectar si el usuario quiere recomendaciones iniciales
        if re.search(r"(?i)s[ií]|claro|por favor|recomien", message) and not self.previous_recommendation_request:
            self.previous_recommendation_request = True
            # Reiniciar las recomendaciones para comenzar de nuevo
            self.recommender.reset_recommendations()
            
            specialists = self.recommender.get_specialists_for_conditions(
                self.qa_system.last_conditions,
                exclude_previous=False
            )
            
            if specialists:
                response = "Te recomendaría consultar con los siguientes especialistas:\n"
                for specialty, reason in specialists.items():
                    response += f"- {specialty}: {reason}\n"
                return response
            else:
                return "Para tus síntomas, te recomendaría inicialmente consultar con un médico general que pueda evaluarte y referirte al especialista adecuado si es necesario."
                
        # Detectar si el usuario está pidiendo alternativas o más opciones
        elif re.search(r"(?i)otro|más|alternativa|diferente|adicional|distinto", message):
            self.previous_recommendation_request = True  # Mantener activo el contexto de recomendación
            
            # Obtener especialistas excluyendo los anteriores
            specialists = self.recommender.get_specialists_for_conditions(
                self.qa_system.last_conditions,
                exclude_previous=True
            )
            
            if specialists:
                response = "También podrías considerar consultar con:\n"
                for specialty, reason in specialists.items():
                    response += f"- {specialty}: {reason}\n"
                return response
            else:
                # Si no hay más especialistas disponibles
                self.recommender.reset_recommendations()  # Reiniciar para futuras consultas
                return "No tengo más recomendaciones específicas de especialistas para tus síntomas actuales. Si tienes otros síntomas o preocupaciones que no has mencionado, házmelo saber para poder aconsejarte mejor."
        else:
            self.conversation_stage = "symptom_collection"
            self.previous_recommendation_request = False
            return "Entiendo. ¿Hay algún otro síntoma que quieras mencionarme?"

    def _generate_symptom_follow_up(self, symptoms):
        """Genera preguntas de seguimiento específicas basadas en los síntomas ya mencionados"""
        if not symptoms:
            return "¿Podrías proporcionarme más detalles sobre cómo te sientes?"
        
        # Verificar si hay texto de prompt en la lista de síntomas y eliminar cualquier texto no deseado
        cleaned_symptoms = []
        for symptom in symptoms:
            if "list of symptoms" in symptom.lower() or "separated by" in symptom.lower():
                continue
            # Eliminar cualquier instrucción o texto no deseado
            if ":" in symptom:
                parts = symptom.split(":")
                if len(parts) > 1 and len(parts[1].strip()) > 0:
                    cleaned_symptoms.append(parts[1].strip())
            else:
                cleaned_symptoms.append(symptom)
        
        # Si después de limpieza no quedan síntomas válidos, hacer pregunta genérica
        if not cleaned_symptoms:
            return "¿Podrías describir con más detalle los síntomas que estás experimentando?"
        
        # Usar los síntomas limpios para el resto de la función
        symptoms = cleaned_symptoms
        
        # Seleccionar un síntoma para hacer seguimiento
        primary_symptom = symptoms[0]
        
        # Preguntas específicas según el síntoma principal
        if "dolor" in primary_symptom:
            return f"Entiendo que tienes {primary_symptom}. ¿Podrías decirme hace cuánto tiempo comenzó, qué tan intenso es y si algo lo mejora o empeora?"
        
        elif "cabeza" in primary_symptom:
            return f"Sobre tu {primary_symptom}, ¿es un dolor pulsátil o constante? ¿Afecta alguna parte específica de la cabeza? ¿Tienes otros síntomas como náuseas o sensibilidad a la luz?"
        
        elif "fiebre" in primary_symptom:
            return f"¿Has medido tu temperatura? ¿Desde cuándo tienes fiebre y va acompañada de escalofríos o sudoración?"
        
        elif "tos" in primary_symptom:
            return f"Respecto a tu tos, ¿es seca o con flema? ¿Has notado si expulsas mucosidad o sangre al toser?"
        
        elif "estómago" in primary_symptom or "abdominal" in primary_symptom:
            return f"¿En qué parte específica del abdomen sientes malestar? ¿Has tenido cambios en tus hábitos intestinales, náuseas o vómitos?"
        
        elif "respirar" in primary_symptom or "aire" in primary_symptom:
            return f"¿La dificultad para respirar es constante o aparece con el esfuerzo? ¿Has notado si empeora al acostarte?"
        
        elif "mareo" in primary_symptom or "vértigo" in primary_symptom:
            return f"¿Cómo describirías exactamente la sensación de mareo? ¿Sientes que el entorno gira, que tú giras, o es más bien como si fueras a desmayarte?"
        
        elif "oído" in primary_symptom or "oreja" in primary_symptom:
            return f"¿El dolor de oído es en uno o ambos oídos? ¿Sientes también zumbidos, pérdida de audición o has tenido alguna secreción?"
        
        elif "muscular" in primary_symptom:
            return f"¿El dolor muscular está localizado en alguna zona específica o es generalizado? ¿Has realizado alguna actividad física intensa recientemente?"
        
        # Pregunta genérica para otros síntomas
        symptom_text = ", ".join(symptoms) if len(symptoms) > 1 else symptoms[0]
        return f"Entiendo que tienes {symptom_text}. ¿Desde cuándo los presentas? ¿Has notado algún otro síntoma o factor que parezca desencadenarlos o aliviarlos?"
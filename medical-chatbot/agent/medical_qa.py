from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from models.qa_model import get_medical_qa_model
from utils.vector_store import get_vector_store

class MedicalQASystem:
    def __init__(self):
        self.llm = get_medical_qa_model()
        self.vector_store = get_vector_store()
        self.last_conditions = []
        
        # Prompts para diferentes tareas
        self.symptom_extraction_prompt = PromptTemplate(
            input_variables=["patient_message"],
            template="""
            Extrae los síntomas mencionados en el siguiente mensaje del paciente:
            
            Mensaje: {patient_message}
            
            Lista de síntomas (separados por coma):
            """
        )
        
        self.condition_analysis_prompt = PromptTemplate(
            input_variables=["symptoms", "context"],
            template="""
            Basado en los siguientes síntomas y la información médica proporcionada,
            lista hasta 3 posibles condiciones médicas que podrían estar relacionadas.
            Escribe solo el nombre de cada condición, una por línea, sin numeración ni prefijos.
            
            Síntomas: {symptoms}
            
            Información médica: {context}
            
            Posibles condiciones médicas:
            """
        )
        
    async def extract_symptoms(self, message):
        chain = LLMChain(llm=self.llm, prompt=self.symptom_extraction_prompt)
        result = await chain.arun(patient_message=message)
        
        # Limpiar el resultado para eliminar cualquier texto de instrucción
        if "lista de síntomas" in result.lower() or "list of symptoms" in result.lower():
            # Extraer solo la parte después de los dos puntos si existe
            parts = result.split(":")
            if len(parts) > 1:
                result = parts[-1].strip()
        
        # Procesar los síntomas extraídos
        symptoms = [s.strip() for s in result.split(",") if s.strip()]
        
        # Filtro adicional para eliminar textos que no son síntomas
        filtered_symptoms = []
        for symptom in symptoms:
            # Ignorar instrucciones o textos del prompt que pudieran haber sido incluidos
            if "lista" in symptom.lower() or "separados" in symptom.lower() or "list" in symptom.lower():
                continue
            # Ignorar fragmentos muy cortos o que no son síntomas reconocibles
            if len(symptom) > 2 and not symptom.isdigit():
                filtered_symptoms.append(symptom)
        
        return filtered_symptoms
    
    async def get_possible_conditions(self, symptoms):
        # Sistema de respaldo con condiciones predefinidas para síntomas comunes
        predefined_conditions = {
            # Síntomas generales
            "fiebre": ["Gripe", "Infección viral", "COVID-19", "Infección bacteriana", "Neumonía"],
            "escalofríos": ["Gripe", "Infección", "Malaria", "Septicemia", "Neumonía"],
            "fatiga": ["Anemia", "Hipotiroidismo", "Depresión", "Mononucleosis", "Apnea del sueño"],
            "cansancio": ["Anemia", "Hipotiroidismo", "Depresión", "Deficiencia de vitaminas", "Enfermedad cardíaca"],
            "debilidad": ["Anemia", "Hipoglucemia", "Miastenia gravis", "Enfermedad de Parkinson", "Esclerosis múltiple"],
            "pérdida de peso": ["Hipertiroidismo", "Diabetes", "Cáncer", "Enfermedad inflamatoria intestinal", "Depresión"],
            "aumento de peso": ["Hipotiroidismo", "Síndrome de Cushing", "Efecto secundario de medicamentos", "Retención de líquidos", "Obesidad"],
            "sudoración nocturna": ["Tuberculosis", "Linfoma", "Infección", "Menopausia", "Apnea del sueño"],
            "malestar general": ["Infección viral", "Gripe", "Reacción a medicamentos", "Estrés", "Fatiga crónica"],
            
            # Síntomas respiratorios
            "tos": ["Resfriado común", "Bronquitis", "Asma", "Neumonía", "COVID-19"],
            "tos seca": ["COVID-19", "Asma", "Alergias", "Reflujo gastroesofágico", "Infección viral"],
            "tos con flema": ["Bronquitis", "Neumonía", "EPOC", "Infección sinusal", "Tuberculosis"],
            "tos con sangre": ["Tuberculosis", "Neumonía", "Cáncer de pulmón", "Embolia pulmonar", "Bronquiectasia"],
            "dificultad para respirar": ["Asma", "Neumonía", "Insuficiencia cardíaca", "EPOC", "Ansiedad"],
            "respiración rápida": ["Asma", "Neumonía", "Ansiedad", "Acidosis metabólica", "Embolia pulmonar"],
            "dolor al respirar": ["Neumonía", "Pleuresía", "Costocondritis", "Embolia pulmonar", "Neumotórax"],
            "congestión nasal": ["Resfriado común", "Sinusitis", "Rinitis alérgica", "Pólipos nasales", "Desviación del tabique"],
            "secreción nasal": ["Resfriado común", "Rinitis alérgica", "Sinusitis", "Cambios de temperatura", "Exposición a irritantes"],
            "estornudos": ["Alergia", "Resfriado común", "Rinitis", "Irritantes ambientales", "Infección viral"],
            "dolor de garganta": ["Faringitis", "Resfriado común", "Amigdalitis", "Laringitis", "Reflujo gastroesofágico"],
            "ronquera": ["Laringitis", "Nódulos vocales", "Cáncer de laringe", "Reflujo", "Hipotiroidismo"],
            "sibilancias": ["Asma", "Bronquitis", "EPOC", "Reacción alérgica", "Insuficiencia cardíaca"],
            
            # Síntomas cardíacos
            "dolor de pecho": ["Angina de pecho", "Infarto de miocardio", "Ansiedad", "Costocondritis", "Reflujo gastroesofágico"],
            "dolor en el pecho": ["Angina de pecho", "Infarto de miocardio", "Ansiedad", "Costocondritis", "Embolia pulmonar"],
            "palpitaciones": ["Arritmia cardíaca", "Ansiedad", "Hipertiroidismo", "Anemia", "Efectos de cafeína"],
            "ritmo cardíaco irregular": ["Fibrilación auricular", "Aleteo auricular", "Extrasístoles", "Enfermedad de la válvula cardíaca", "Cardiomiopatía"],
            "presión arterial alta": ["Hipertensión esencial", "Enfermedad renal", "Apnea del sueño", "Síndrome de Cushing", "Feocromocitoma"],
            "presión arterial baja": ["Deshidratación", "Hemorragia", "Septicemia", "Medicamentos", "Insuficiencia suprarrenal"],
            "desmayo": ["Síncope vasovagal", "Hipotensión ortostática", "Arritmia cardíaca", "Hipoglucemia", "Anemia"],
            "hinchazón de piernas": ["Insuficiencia cardíaca", "Insuficiencia venosa", "Trombosis venosa profunda", "Insuficiencia renal", "Cirrosis"],
            
            # Síntomas digestivos
            "dolor abdominal": ["Gastritis", "Apendicitis", "Cólico biliar", "Pancreatitis", "Enfermedad inflamatoria intestinal"],
            "dolor estomacal": ["Gastritis", "Úlcera péptica", "Reflujo gastroesofágico", "Dispepsia funcional", "Cáncer gástrico"],
            "náuseas": ["Gastroenteritis", "Migraña", "Embarazo", "Intoxicación alimentaria", "Efectos secundarios de medicamentos"],
            "vómitos": ["Gastroenteritis", "Intoxicación alimentaria", "Obstrucción intestinal", "Migraña", "Apendicitis"],
            "diarrea": ["Gastroenteritis", "Intoxicación alimentaria", "Síndrome de intestino irritable", "Enfermedad de Crohn", "Colitis ulcerosa"],
            "estreñimiento": ["Dieta baja en fibra", "Deshidratación", "Síndrome de intestino irritable", "Hipotiroidismo", "Efectos secundarios de medicamentos"],
            "heces negras": ["Sangrado gastrointestinal superior", "Uso de hierro oral", "Bismuto", "Cáncer colorrectal", "Úlcera péptica"],
            "sangre en las heces": ["Hemorroides", "Fisuras anales", "Enfermedad inflamatoria intestinal", "Pólipos colorectales", "Cáncer colorrectal"],
            "acidez": ["Reflujo gastroesofágico", "Hernia hiatal", "Gastritis", "Úlcera péptica", "Embarazo"],
            "distensión abdominal": ["Síndrome de intestino irritable", "Intolerancia a lactosa", "Enfermedad celíaca", "Ascitis", "Obstrucción intestinal"],
            "ictericia": ["Hepatitis", "Cirrosis", "Obstrucción biliar", "Anemia hemolítica", "Cáncer de páncreas"],
            "dificultad para tragar": ["Enfermedad por reflujo gastroesofágico", "Acalasia", "Cáncer de esófago", "Ansiedad", "Esclerosis lateral amiotrófica"],
            
            # Síntomas neurológicos
            "dolor de cabeza": ["Migraña", "Cefalea tensional", "Sinusitis", "Hipertensión", "Tumor cerebral"],
            "cefalea": ["Migraña", "Cefalea tensional", "Cefalea en racimos", "Meningitis", "Aneurisma cerebral"],
            "mareo": ["Vértigo", "Hipotensión", "Anemia", "Deshidratación", "Ansiedad"],
            "vértigo": ["Vértigo posicional paroxístico benigno", "Enfermedad de Ménière", "Neuritis vestibular", "Laberintitis", "Tumor cerebral"],
            "entumecimiento": ["Neuropatía periférica", "Compresión nerviosa", "Esclerosis múltiple", "Accidente cerebrovascular", "Diabetes"],
            "hormigueo": ["Neuropatía periférica", "Deficiencia de vitamina B12", "Síndrome del túnel carpiano", "Esclerosis múltiple", "Migraña con aura"],
            "debilidad muscular": ["Esclerosis múltiple", "Miastenia gravis", "Polimiositis", "Enfermedad de Parkinson", "Esclerosis lateral amiotrófica"],
            "temblores": ["Enfermedad de Parkinson", "Temblor esencial", "Efectos secundarios de medicamentos", "Abstinencia de alcohol", "Hipertiroidismo"],
            "confusión": ["Delirium", "Demencia", "Infección", "Efectos secundarios de medicamentos", "Encefalopatía metabólica"],
            "problemas de memoria": ["Enfermedad de Alzheimer", "Demencia vascular", "Depresión", "Hipotiroidismo", "Deficiencia de vitamina B12"],
            "convulsiones": ["Epilepsia", "Abstinencia de alcohol", "Hipoglucemia", "Fiebre alta", "Tumor cerebral"],
            "dificultad para hablar": ["Accidente cerebrovascular", "Esclerosis lateral amiotrófica", "Enfermedad de Parkinson", "Distonía", "Ansiedad"],
            "parálisis facial": ["Parálisis de Bell", "Accidente cerebrovascular", "Síndrome de Guillain-Barré", "Enfermedad de Lyme", "Tumor cerebral"],
            
            # Síntomas musculoesqueléticos
            "dolor articular": ["Artritis", "Artrosis", "Gota", "Lupus", "Enfermedad de Lyme"],
            "dolor muscular": ["Fibromialgia", "Polimialgia reumática", "Rabdomiólisis", "Infección viral", "Efectos secundarios de estatinas"],
            "dolor de espalda": ["Hernia de disco", "Estenosis espinal", "Osteoartritis", "Fibromialgia", "Enfermedad renal"],
            "dolor lumbar": ["Tensión muscular", "Hernia de disco", "Estenosis espinal", "Enfermedad degenerativa de disco", "Espondilolistesis"],
            "rigidez articular": ["Artritis reumatoide", "Artrosis", "Fibromialgia", "Lupus", "Polimialgia reumática"],
            "hinchazón articular": ["Artritis", "Gota", "Bursitis", "Lupus", "Lesión traumática"],
            "dolor de cuello": ["Tensión muscular", "Hernia de disco cervical", "Espondilosis cervical", "Fibromialgia", "Meningitis"],
            "dolor en extremidades": ["Neuropatía periférica", "Enfermedad arterial periférica", "Trombosis venosa profunda", "Fibromialgia", "Poliomiositis"],
            
            # Síntomas de la piel
            "erupción cutánea": ["Dermatitis", "Urticaria", "Psoriasis", "Infección fúngica", "Reacción alérgica"],
            "picazón": ["Dermatitis", "Urticaria", "Psoriasis", "Escabiosis", "Reacción alérgica"],
            "enrojecimiento de la piel": ["Dermatitis", "Rosácea", "Quemadura solar", "Celulitis", "Lupus"],
            "ampollas": ["Herpes", "Impétigo", "Quemaduras", "Dermatitis de contacto", "Reacciones medicamentosas"],
            "cambios en lunares": ["Melanoma", "Carcinoma basocelular", "Carcinoma epidermoide", "Queratosis seborreica", "Nevus displásico"],
            "urticaria": ["Alergia alimentaria", "Alergia a medicamentos", "Infección", "Estrés", "Exposición al calor o frío"],
            "piel seca": ["Dermatitis atópica", "Psoriasis", "Hipotiroidismo", "Deshidratación", "Deficiencia nutricional"],
            "sudoración excesiva": ["Hipertiroidismo", "Ansiedad", "Infección", "Obesidad", "Medicamentos"],
            
            # Síntomas oculares
            "visión borrosa": ["Error refractivo", "Cataratas", "Glaucoma", "Retinopatía diabética", "Degeneración macular"],
            "ojos rojos": ["Conjuntivitis", "Uveítis", "Glaucoma", "Ojo seco", "Blefaritis"],
            "dolor ocular": ["Glaucoma", "Uveítis", "Sinusitis", "Migraña", "Conjuntivitis"],
            "sensibilidad a la luz": ["Migraña", "Meningitis", "Uveítis", "Quemadura corneal", "Ojo seco"],
            "visión doble": ["Miastenia gravis", "Esclerosis múltiple", "Accidente cerebrovascular", "Trauma craneal", "Aneurisma"],
            "pérdida de visión": ["Glaucoma", "Accidente cerebrovascular", "Desprendimiento de retina", "Neuritis óptica", "Oclusión arterial retiniana"],
            "ojos secos": ["Síndrome de ojo seco", "Síndrome de Sjögren", "Blefaritis", "Deficiencia de vitamina A", "Efectos secundarios de medicamentos"],
            "ceguera nocturna": ["Deficiencia de vitamina A", "Retinitis pigmentosa", "Degeneración macular", "Cataratas", "Glaucoma"],
            
            # Síntomas auditivos
            "pérdida de audición": ["Presbiacusia", "Otosclerosis", "Enfermedad de Ménière", "Trauma acústico", "Infección del oído"],
            "tinnitus": ["Pérdida de audición inducida por ruido", "Enfermedad de Ménière", "Otosclerosis", "Efectos secundarios de medicamentos", "Tumor del nervio acústico"],
            "dolor de oído": ["Otitis media", "Otitis externa", "Infección del oído", "Disfunción de la articulación temporomandibular", "Absceso dental"],
            "secreción del oído": ["Otitis media", "Otitis externa", "Perforación del tímpano", "Colesteatoma", "Trauma"],
            "sensación de oído tapado": ["Disfunción tubárica", "Tapón de cerumen", "Otitis media", "Barotrauma", "Otosclerosis"],
            
            # Síntomas urinarios
            "dolor al orinar": ["Infección urinaria", "Uretritis", "Prostatitis", "Cálculos renales", "Cistitis intersticial"],
            "micción frecuente": ["Infección urinaria", "Diabetes", "Hiperplasia prostática benigna", "Embarazo", "Vejiga hiperactiva"],
            "micción urgente": ["Infección urinaria", "Vejiga hiperactiva", "Cistitis intersticial", "Hiperplasia prostática", "Cáncer de vejiga"],
            "sangre en la orina": ["Infección urinaria", "Cálculos renales", "Cistitis", "Cáncer de vejiga", "Glomerulonefritis"],
            "incontinencia urinaria": ["Hiperplasia prostática benigna", "Vejiga hiperactiva", "Prolapso pélvico", "Efectos secundarios de medicamentos", "Esclerosis múltiple"],
            "disminución del flujo urinario": ["Hiperplasia prostática benigna", "Estenosis uretral", "Cáncer de próstata", "Vejiga neurogénica", "Infección urinaria"],
            "orina oscura": ["Deshidratación", "Hepatitis", "Rabdomiólisis", "Anemia hemolítica", "Porfiria"],
            
            # Síntomas psicológicos
            "ansiedad": ["Trastorno de ansiedad generalizada", "Trastorno de pánico", "Fobia social", "Estrés postraumático", "Hipertiroidismo"],
            "depresión": ["Trastorno depresivo mayor", "Trastorno bipolar", "Distimia", "Hipotiroidismo", "Trastorno afectivo estacional"],
            "insomnio": ["Ansiedad", "Depresión", "Apnea del sueño", "Síndrome de piernas inquietas", "Efectos secundarios de medicamentos"],
            "cambios de humor": ["Trastorno bipolar", "Trastorno premenstrual", "Depresión", "Menopausia", "Trastorno de personalidad límite"],
            "irritabilidad": ["Ansiedad", "Depresión", "Trastorno bipolar", "Hipertiroidismo", "Síndrome premenstrual"],
            "pensamientos suicidas": ["Depresión mayor", "Trastorno bipolar", "Esquizofrenia", "Trastorno de estrés postraumático", "Trastorno de personalidad límite"],
            "alucinaciones": ["Esquizofrenia", "Trastorno bipolar", "Demencia", "Intoxicación por drogas", "Delirium"],
            "paranoia": ["Esquizofrenia", "Trastorno delirante", "Demencia", "Intoxicación por drogas", "Trastorno de personalidad paranoide"],
            "ataques de pánico": ["Trastorno de pánico", "Fobia específica", "Trastorno de ansiedad social", "Hipertiroidismo", "Prolapso de la válvula mitral"],
            
            # Síntomas endocrinos
            "sed excesiva": ["Diabetes mellitus", "Diabetes insípida", "Hipertiroidismo", "Deshidratación", "Medicamentos"],
            "hambre excesiva": ["Diabetes mellitus", "Hipertiroidismo", "Hipoglucemia", "Medicamentos", "Bulimia nerviosa"],
            "intolerancia al calor": ["Hipertiroidismo", "Menopausia", "Medicamentos", "Lesión hipotalámica", "Feocromocitoma"],
            "intolerancia al frío": ["Hipotiroidismo", "Anemia", "Enfermedad de Raynaud", "Desnutrición", "Falta de grasa corporal"],
            "cambios en la distribución del vello": ["Hirsutismo", "Síndrome de ovario poliquístico", "Hiperplasia suprarrenal congénita", "Tumores productores de andrógenos", "Medicamentos"],
            
            # Síntomas reproductivos y sexuales
            "disfunción eréctil": ["Enfermedad cardiovascular", "Diabetes", "Hipertensión", "Depresión", "Efectos secundarios de medicamentos"],
            "disminución del deseo sexual": ["Depresión", "Bajo nivel de testosterona", "Estrés", "Efectos secundarios de medicamentos", "Problemas de relación"],
            "dolor durante las relaciones sexuales": ["Vaginismo", "Endometriosis", "Infección vaginal", "Sequedad vaginal", "Prostatitis"],
            "sangrado vaginal anormal": ["Pólipos uterinos", "Fibromas", "Cáncer de endometrio", "Desequilibrio hormonal", "Endometriosis"],
            "dolor menstrual": ["Endometriosis", "Adenomiosis", "Enfermedad inflamatoria pélvica", "Fibromas", "Síndrome premenstrual"],
            "flujo vaginal anormal": ["Vaginosis bacteriana", "Candidiasis", "Tricomoniasis", "Clamidia", "Gonorrea"],
            "bulto en los senos": ["Quiste mamario", "Fibroadenoma", "Cáncer de mama", "Cambios fibroquísticos", "Mastitis"],
            "secreción del pezón": ["Papiloma intraductal", "Cambios fibroquísticos", "Cáncer de mama", "Medicamentos", "Desequilibrio hormonal"],
            
            # Síntomas específicos
            "fiebre alta": ["Infección bacteriana", "Neumonía", "Meningitis", "Septicemia", "Malaria"],
            "deshidratación": ["Gastroenteritis", "Diarrea", "Vómitos", "Golpe de calor", "Diabetes descontrolada"],
            "somnolencia excesiva": ["Apnea del sueño", "Narcolepsia", "Depresión", "Hipotiroidismo", "Deficiencia de vitamina B12"],
            "dificultad para concentrarse": ["TDAH", "Ansiedad", "Depresión", "Trastorno del sueño", "Efecto secundario de medicamentos"],
            "ronquidos": ["Apnea del sueño", "Obesidad", "Pólipos nasales", "Desviación del tabique", "Consumo de alcohol"],
            "pérdida del gusto": ["COVID-19", "Resfriado común", "Sinusitis", "Medicamentos", "Deficiencia de zinc"],
            "pérdida del olfato": ["COVID-19", "Resfriado común", "Sinusitis", "Pólipos nasales", "Enfermedad de Parkinson"],
            "dolor dental": ["Caries", "Absceso dental", "Gingivitis", "Periodontitis", "Sensibilidad dental"],
            "sangrado de encías": ["Gingivitis", "Periodontitis", "Trastornos de la coagulación", "Leucemia", "Escorbuto"],
            "rigidez de cuello": ["Meningitis", "Tensión muscular", "Artritis cervical", "Fibromialgia", "Tortícolis"],
            "piernas inquietas": ["Síndrome de piernas inquietas", "Deficiencia de hierro", "Embarazo", "Insuficiencia renal", "Neuropatía"],
            "calambres musculares": ["Deshidratación", "Desequilibrio electrolítico", "Deficiencia de magnesio", "Medicamentos", "Síndrome de piernas inquietas"],
            "inflamación de los ganglios linfáticos": ["Infección", "Mononucleosis", "Trastornos autoinmunes", "Cáncer", "VIH/SIDA"],
            "tos al acostarse": ["Reflujo gastroesofágico", "Insuficiencia cardíaca", "Asma", "Bronquitis", "Goteo posnasal"],
            "dolor en las pantorrillas": ["Trombosis venosa profunda", "Calambres musculares", "Shin splints", "Insuficiencia venosa", "Claudicación intermitente"],
            "decoloración de la piel": ["Vitiligo", "Hígado graso", "Anemia", "Enfermedad de Addison", "Insuficiencia renal"],
            "bultos en el cuello": ["Aumento de ganglios linfáticos", "Bocio", "Quiste tirogloso", "Cáncer de tiroides", "Lipoma"],
            "falta de aliento al acostarse": ["Insuficiencia cardíaca", "EPOC", "Asma", "Ansiedad", "Obesidad"],
            "confusión repentina": ["Accidente cerebrovascular", "Ataque isquémico transitorio", "Infección", "Hipoglucemia", "Delirium"]
        }

        # Verificar si alguno de los síntomas está en nuestro diccionario predefinido
        matched_conditions = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, conditions in predefined_conditions.items():
                if key in symptom_lower or symptom_lower in key:
                    matched_conditions.extend(conditions)
                    
        # Si encontramos condiciones predefinidas, usarlas
        if matched_conditions:
            # Eliminar duplicados y limitar a 3 condiciones
            unique_conditions = list(dict.fromkeys(matched_conditions))
            self.last_conditions = unique_conditions[:3]
            return self.last_conditions
        
        # Respuesta genérica de respaldo 
        default_conditions = ["Posible afección leve", "Condición temporal", "Reacción al estrés"]
        self.last_conditions = default_conditions
        return default_conditions
    
    async def answer_medical_question(self, question, memory):
        # Recuperar documentos relevantes para la pregunta
        docs = self.vector_store.similarity_search(
            question,
            k=3
        )
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prompt para responder preguntas médicas
        medical_qa_prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
            Como asistente médico virtual, responde a la siguiente pregunta utilizando
            la información proporcionada y el historial de chat. Si no estás seguro o 
            la pregunta requiere diagnóstico médico profesional, indícalo claramente.
            
            Historial: {chat_history}
            
            Información médica: {context}
            
            Pregunta: {question}
            
            Respuesta:
            """
        )
        
        # Obtener historial de chat
        chat_history = memory.load_memory_variables({})["history"]
        
        # Crear y ejecutar la cadena
        chain = LLMChain(llm=self.llm, prompt=medical_qa_prompt)
        response = await chain.arun(
            question=question,
            context=context,
            chat_history=chat_history
        )
        
        return response
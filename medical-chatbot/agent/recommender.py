import json
import os
import random

class SpecialistRecommender:
    def __init__(self):
        self.specialists_data = self._load_specialists_data()
        self.recommended_specialists = set()  # Para rastrear especialistas ya recomendados
        
        # Mapeo de condiciones a especialidades médicas (principal y alternativas)
        self.condition_specialties = {
            # Condiciones generales (para fallbacks)
            "posible afección": ["Medicina General", "Medicina Interna", "Medicina Familiar"],
            "posible afección leve": ["Medicina General", "Medicina Familiar", "Medicina Preventiva"],
            "condición temporal": ["Medicina General", "Medicina Familiar", "Medicina Interna"],
            "afección temporal": ["Medicina General", "Medicina Familiar", "Medicina Preventiva"],
            "estrés": ["Psicología", "Psiquiatría", "Medicina General"],
            "reacción al estrés": ["Psicología", "Psiquiatría", "Medicina General"],
            "condición leve": ["Medicina General", "Medicina Familiar", "Medicina Preventiva"],
            
            # Condiciones específicas
            "gripe": ["Medicina General", "Neumología", "Infectología"],
            "resfriado": ["Medicina General", "Neumología", "Otorrinolaringología"],
            "covid-19": ["Neumología", "Medicina General", "Infectología"],
            "hipertensión": ["Cardiología", "Medicina General", "Nefrología"],
            "diabetes": ["Endocrinología", "Medicina General", "Nutrición"],
            "ansiedad": ["Psiquiatría", "Psicología", "Neurología"],
            "depresión": ["Psiquiatría", "Psicología", "Neurología"],
            "artritis": ["Reumatología", "Medicina General", "Traumatología"],
            "alergia": ["Alergología", "Dermatología", "Neumología"],
            "migraña": ["Neurología", "Medicina General", "Medicina del Dolor"],
            "dolor de pecho": ["Cardiología", "Medicina General", "Neumología"],
            "angina": ["Cardiología", "Medicina General", "Medicina de Emergencia"],
            "infarto": ["Cardiología", "Medicina de Emergencia", "Medicina Intensiva"],
            "reflujo": ["Gastroenterología", "Medicina General", "Otorrinolaringología"],
            "bronquitis": ["Neumología", "Medicina General", "Alergología"],
            "asma": ["Neumología", "Alergología", "Medicina General"],
            "vértigo": ["Otorrinolaringología", "Neurología", "Medicina General"],
            "gastritis": ["Gastroenterología", "Medicina General", "Medicina Interna"],
            "apendicitis": ["Cirugía General", "Medicina de Emergencia", "Medicina General"],
        }
        
        # Lista de especialistas genéricos para cuando no hay coincidencias específicas
        self.generic_specialists = [
            "Medicina General", 
            "Medicina Interna", 
            "Medicina Familiar", 
            "Medicina Preventiva",
            "Medicina Integral",
            "Clínica General"
        ]
    
    def _load_specialists_data(self):
        """Carga datos de especialistas desde JSON"""
        try:
            with open("data/specialists.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si el archivo no existe o está mal formateado, usar datos predeterminados
            return {
                "Medicina General": {
                    "description": "Médicos generales que pueden tratar una amplia variedad de condiciones",
                    "when_to_see": "Síntomas generales, chequeos anuales, y referencia a especialistas"
                },
                "Medicina Interna": {
                    "description": "Especialistas en diagnóstico y tratamiento de adultos",
                    "when_to_see": "Condiciones complejas, múltiples síntomas, manejo de enfermedades crónicas"
                },
                "Medicina Familiar": {
                    "description": "Médicos que atienden a toda la familia",
                    "when_to_see": "Atención integral para todas las edades, seguimiento a largo plazo"
                },
                "Medicina Preventiva": {
                    "description": "Especialistas en prevención de enfermedades",
                    "when_to_see": "Evaluaciones de riesgo, chequeos preventivos, consejos de salud"
                },
                "Cardiología": {
                    "description": "Especialistas en enfermedades del corazón y sistema circulatorio",
                    "when_to_see": "Problemas cardíacos, hipertensión, dolor en el pecho"
                },
                "Neurología": {
                    "description": "Especialistas en el sistema nervioso",
                    "when_to_see": "Dolores de cabeza crónicos, mareos, problemas de memoria"
                },
                "Psiquiatría": {
                    "description": "Especialistas en salud mental",
                    "when_to_see": "Depresión, ansiedad, trastornos del sueño"
                },
                "Psicología": {
                    "description": "Profesionales de la salud mental no médicos",
                    "when_to_see": "Problemas emocionales, estrés, terapia de comportamiento"
                },
                "Neumología": {
                    "description": "Especialistas en el sistema respiratorio",
                    "when_to_see": "Problemas respiratorios, asma, EPOC, neumonía"
                },
                "Gastroenterología": {
                    "description": "Especialistas en el sistema digestivo",
                    "when_to_see": "Problemas digestivos, reflujo, úlceras, enfermedades intestinales"
                },
                "Endocrinología": {
                    "description": "Especialistas en el sistema endocrino",
                    "when_to_see": "Diabetes, problemas de tiroides, trastornos hormonales"
                },
                "Dermatología": {
                    "description": "Especialistas en la piel",
                    "when_to_see": "Problemas de piel, alergias cutáneas, acné, lunares"
                },
                "Otorrinolaringología": {
                    "description": "Especialistas en oído, nariz y garganta",
                    "when_to_see": "Problemas de oído, sinusitis, vértigo, problemas de voz"
                },
                "Traumatología": {
                    "description": "Especialistas en el sistema músculo-esquelético",
                    "when_to_see": "Lesiones, fracturas, problemas articulares"
                },
            }
    
    def get_specialists_for_conditions(self, conditions, exclude_previous=False):
        """
        Retorna especialistas recomendados basados en las condiciones posibles
        """
        recommendations = {}
        potential_specialists = []
        
        # Recopilar todos los posibles especialistas para las condiciones
        for condition in conditions:
            condition_normalized = condition.lower()
            
            # Buscar coincidencias en el diccionario de condiciones-especialidades
            for known_condition, specialties in self.condition_specialties.items():
                if known_condition in condition_normalized or condition_normalized in known_condition:
                    # Añadir especialistas con su prioridad (basada en el orden en la lista)
                    for i, specialty in enumerate(specialties):
                        if specialty in self.specialists_data:
                            # Menor valor = mayor prioridad
                            potential_specialists.append((specialty, i))
        
        # Si no hay coincidencias, usar lista de especialistas genéricos
        if not potential_specialists:
            for i, specialty in enumerate(self.generic_specialists):
                if specialty in self.specialists_data:
                    potential_specialists.append((specialty, i))
        
        # Ordenar por prioridad y eliminar duplicados manteniendo la prioridad más alta
        sorted_specialists = sorted(potential_specialists, key=lambda x: x[1])
        unique_specialists = []
        seen = set()
        for specialty, _ in sorted_specialists:
            if specialty not in seen:
                unique_specialists.append(specialty)
                seen.add(specialty)
        
        # Filtrar especialistas ya recomendados si es necesario
        available_specialists = unique_specialists
        if exclude_previous:
            available_specialists = [s for s in available_specialists if s not in self.recommended_specialists]
        
        # Si no quedan especialistas disponibles, ofrecer alternativas que no estén en el mapeo
        if not available_specialists:
            alternative_specialists = [s for s in self.specialists_data.keys() 
                                     if s not in self.recommended_specialists]
            
            # Si aún no hay alternativas, reiniciar el seguimiento
            if not alternative_specialists:
                self.reset_recommendations()
                # Excepto Medicina General si ya fue recomendada, para evitar repeticiones
                if "Medicina General" in self.specialists_data:
                    self.recommended_specialists.add("Medicina General")
                return self.get_specialists_for_conditions(conditions, True)
            
            # Seleccionar especialistas alternativos
            available_specialists = alternative_specialists
        
        # Seleccionar hasta 2 especialistas (o todos si hay menos de 2)
        selected_count = min(2, len(available_specialists))
        if selected_count > 0:
            selected_specialists = available_specialists[:selected_count]  # Ya están ordenados por prioridad
            
            # Crear respuesta con los especialistas seleccionados
            for specialty in selected_specialists:
                self.recommended_specialists.add(specialty)  # Marcar como recomendado
                recommendations[specialty] = self.specialists_data[specialty]["when_to_see"]
        
        return recommendations
    
    def reset_recommendations(self):
        """Reinicia el registro de especialistas recomendados"""
        self.recommended_specialists.clear()
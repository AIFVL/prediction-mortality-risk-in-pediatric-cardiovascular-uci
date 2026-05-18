# Predicción del riesgo de mortalidad en UCI cardiovascular en pacientes pediátricos con cardiopatías congénitas mediante modelos de inteligencia artificial

----

> Integrantes:
>Pablo Fernando Pineda Patiño-> [Pabl0FPP](https://github.com/Pabl0FPP)

>Jennifer Castro Cadena -> [JenniferCastrocd](https://github.com/JenniferCastrocd)

>Rafaela Sofia Ruiz Pizarro -> [RafaelaRuiz](https://github.com/RafaelaRuiz)
## 1. Introducción

### 1.1 Contexto
Las cardiopatías congénitas (CC) constituyen una de las principales causas de ingreso a unidades de cuidado intensivo cardiovascular (UCIC) en pacientes pediátricos y representan un desafío crítico para la salud pública, especialmente en países como Colombia. Según reportes oficiales recientes del instituto Nacional de salud (2025), la prevalencia de cardiopatías congénitas en Colombia es de aproximadamente 79.9 cada 10,000 nacidos vivos, con una alta mortalidad asociada en el primer año de vida debido a la complejidad de estas enfermedades y la necesidad de intervenciones quirúrgicas especializadas.

Actualmente, la estimación del riesgo de mortalidad asociada a procedimientos quirúrgicos cardiovascular en pediatría se realiza principalmente mediante la escala Risk Adjustment for Congenital Heart Surgery (RACHS-1). Aunque ampliamente utilizada, la escala presenta limitaciones importantes, ya que se enfoca únicamente en el tipo de procedimiento quirúrgico sin tomar en cuenta variables individuales del paciente.

### 1.2 Formulación del problema
En la Fundación Valle del Lili, la mortalidad en pacientes pediátricos con cardiopatías congénitas que requieren ingreso a la UCIC tras cirugía sigue siendo un reto clínico por la dificultad de predecir con precisión los resultados postoperatorios. Las herramientas tradicionales no logran integrar la complejidad clínica individual ni la diversidad de variables que influyen en el pronóstico.

### 1.3 Impacto del proyecto
La aplicación de modelos de inteligencia artificial permitiría mejorar la estratificación del riesgo, apoyar decisiones clínicas más oportunas y personalizadas, optimizar el uso de recursos hospitalarios y, en última instancia, aumentar la supervivencia y la calidad de vida de los pacientes pediátricos.

## 2. Objetivos

### 2.1 Objetivo General
Desarrollar un modelo predictivo de inteligencia artificial para estimar el riesgo de mortalidad postoperatoria en pacientes pediátricos con cardiopatías congénitas en la UCI de la Fundación Valle del Lili, con el fin de optimizar la clasificación del riesgo y agilizar la toma de decisiones por parte del personal especializado.

### 2.2 Objetivos Específicos
*   Recolectar y depurar la base de datos histórica de la Fundación Valle del Lili, eliminando registros incompletos y estandarizando variables clínicas.
*   Seleccionar y definir las variables relevantes para el pronóstico de mortalidad postoperatoria en colaboración con especialistas clínicos.
*   Diseñar e implementar el modelo de inteligencia artificial utilizando técnicas de aprendizaje automático, asegurando su reproducibilidad y documentación.
*   Evaluar y validar el desempeño del modelo predictivo frente a escalas tradicionales como RACHS-1.
*   Desarrollar un dashboard interactivo que permita visualizar los resultados del modelo y facilitar la interpretación de los indicadores de riesgo.

---

## 3. Índice de Documentación

Explore los siguientes documentos para entender a fondo la estructura y operación del proyecto:

### 📂 [Estructura del Proyecto](./docs/STRUCTURE.md)
*Contexto sobre la organización del repositorio, descripción de cada carpeta y el propósito de los archivos principales para mantener un entorno modular y escalable.*

### 🚀 [Guía de Configuración y Despliegue (Setup)](./docs/SETUP.md)
*Instrucciones paso a paso para configurar el entorno local, gestión de variables de entorno (`.env`), comandos de ejecución y enlaces a los servicios desplegados en la nube (Vercel y Render).*

### ⚙️ [Flujo del Pipeline y Metodología](./docs/PIPELINE_FLOW.md)
*Detalle de la transición desde los Notebooks exploratorios hacia el Pipeline automatizado. Explica la comunicación entre el Dashboard, la API y el flujo secuencial de los 8 pasos del motor de ML.*

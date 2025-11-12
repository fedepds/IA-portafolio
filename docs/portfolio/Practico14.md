# Práctica UT4-14: LLMs con LangChain (OpenAI) — Prompting, Plantillas y Salida Estructurada

En esta práctica, se abordó la orquestación de Modelos de Lenguaje (LLMs) para construir aplicaciones robustas. El enfoque se centró en pasar de invocaciones simples (`llm.invoke`) a la creación de **cadenas (pipelines) fiables, observables y estructuradas** usando LangChain. Se exploraron los parámetros de decodificación, el uso de plantillas (templates), el encadenamiento con LCEL (`|`), y la crucial habilidad de forzar salidas estructuradas (JSON/Pydantic). Finalmente, se integraron estos conceptos para construir un sistema de RAG (Generación Aumentada por Recuperación) que simula un chatbot de soporte.

## Objetivos
- Instanciar un modelo de chat de OpenAI (`ChatOpenAI`) y controlar parámetros de decodificación (`temperature`, `max_tokens`).
- Diseñar prompts reutilizables con `ChatPromptTemplate` y encadenar componentes usando LangChain Expression Language (LCEL).
- Forzar y garantizar salidas estructuradas (JSON) que adhieran a un esquema `Pydantic` usando `with_structured_output`.
- Utilizar `LangSmith` para medir métricas clave (tokens y latencia) y entender la observabilidad de la cadena.
- Comparar el rendimiento y formato de `zero-shot` vs. `few-shot prompting` y analizar el impacto de la temperatura en ambos.
- Implementar un patrón de `Map-Reduce` para procesar textos que exceden la ventana de contexto.
- Construir un pipeline de RAG básico para "aterrizar" (ground) las respuestas del modelo en una base de conocimiento local (FAISS).

## Metodología

### 1. Control de Parámetros de Decodificación
- **Problema**: Se analizó el efecto de `temperature` en la generación.
- **Técnica**: Se invocó a `ChatOpenAI` con `temperature=0.0` (determinista) y `temperature=0.9` (creativo) (Parte 1).
- **Diagnóstico**: Se observó que la temperatura solo afecta a **tareas abiertas** (creativas, como el tuit) y no a **tareas factuales** (como la identidad del modelo), donde la respuesta es casi idéntica independientemente del parámetro.

### 2. Plantillas y Encadenamiento (LCEL)
- **Objetivo**: Separar la lógica de la instrucción (`prompt`) de los datos (`variables`).
- **Técnica**: Se reemplazaron los strings de prompt por `ChatPromptTemplate` y se encadenó con el LLM usando el operador `|` (LCEL) (Parte 2).
- **Análisis (Zero-shot vs. Few-shot)**: Se compararon los patrones (Parte 6). Se descubrió que un `temperature > 0` podía **romper la precisión** de un modelo *few-shot* si los ejemplos proporcionados estaban incompletos (ej. faltaba la clase `NEU`).

### 3. Garantía de Salida Estructurada (El "Contrato")
- **Hipótesis**: "Pedir" un JSON mediante un prompt es frágil; se debe "forzar" un JSON mediante un esquema.
- **Técnica**: Se definió un esquema `Pydantic` (ej. `Traduccion`, `Resumen`, `ExtractInfo`) y se vinculó al modelo usando `llm.with_structured_output(...)` (Parte 3, 5, 8).
- **Resultado**: Esto eliminó la necesidad de parsear strings, proporcionando un **objeto Python nativo** fiable como salida.

### 4. Observabilidad (Costos y Velocidad)
- **Objetivo**: Medir el rendimiento de la cadena.
- **Técnica**: Se utilizó `LangSmith` (activado en Parte 0) para inspeccionar las trazas de ejecución (Parte 4).
- **Análisis**: Se midió la `latencia` (velocidad) y el `uso de tokens` (costo) de cada invocación, entendiendo el trade-off entre una respuesta rápida (baja latencia) y una respuesta de alta calidad/estructurada (mayor costo y latencia).

### 5. RAG y Desafío Integrador
- **Objetivo**: Construir un chatbot de soporte que responda desde una base de conocimiento local (FAQs) y devuelva un JSON estructurado.
- **Técnica (RAG)**: Se implementó un RAG básico (Parte 9):
  1. **Indexación**: Se dividieron (`RecursiveCharacterTextSplitter`) y vectorizaron (`OpenAIEmbeddings`) documentos en una base `FAISS`.
  2. **Recuperación**: Se creó un `retriever` para buscar los *chunks* relevantes.
- **Técnica (Desafío)**: Se creó una cadena `rag_chain_structured` que combinó el `retriever` (Parte 9) con el `prompt` de *grounding* (Parte 5) y la `salida estructurada` (Parte 8).

## Resultados Principales

### 1. El Peligro de `Few-Shot` + `Temperature`
El análisis de la Parte 6 fue revelador. Mientras que con `T=0` los modelos *zero-shot* y *few-shot* clasificaron correctamente (POS, NEG, **NEU**), al usar `T=0.3` el modelo *few-shot* falló:
- Ejemplo: Texto: "Está bien, nada extraordinario." -> `POS` (Incorrecto).
- El modelo *zero-shot* (con `T=0.3`) siguió acertando (`NEU`).

Conclusión: El *prompt* *few-shot* (que no incluía ejemplos de `NEU`) creó un **sesgo**. La temperatura, al permitir "creatividad", hizo que el modelo siga ese sesgo y falle, demostrando que *few-shot* mal implementado puede ser peor que *zero-shot*.

### 2. Fiabilidad del `with_structured_output`
El uso de `Pydantic` (Partes 3, 5, 8, Desafío) fue 100% fiable. En lugar de devolver un `string` que podría contener un JSON, el modelo devolvió un objeto Python (`Resumen`, `Traduccion`, `RespuestaSoporte`).

Conclusión: Esta técnica es un pilar de producción. Transforma al LLM de un generador de texto a un **componente de software fiable con un contrato de API (el esquema Pydantic)**.

### 3. Éxito del RAG y Manejo de Fallos (Desafío)
El chatbot integrador demostró el éxito del patrón RAG + grounding + salida estructurada:

- Caso 1 (Éxito RAG):
  - Pregunta: "¿Qué significa la luz azul parpadeante?"
  - Respuesta: answer = "La luz azul parpadeante indica que el dispositivo está buscando conexión WiFi."
  - Confianza: confidence = "high"
  - El retriever encontró el chunk correcto, y el LLM lo usó para generar la respuesta.

- Caso 2 (Éxito del "Fallo Controlado"):
  - Pregunta: "¿Cómo cambio la batería?"
  - Respuesta: answer = "No encontré información sobre cómo cambiar la batería..."
  - Confianza: confidence = "low"
  - El retriever no encontró contexto relevante. El LLM, obedeciendo la instrucción "Respondé SOLO con el contexto...", **evitó la alucinación** y reportó el fallo correctamente. Este es un comportamiento deseado y robusto.

## Conclusiones
- **LangChain (LCEL)** permite pasar de prompts a pipelines. El operador `|` es la herramienta clave para diseñar sistemas de IA complejos de forma declarativa.
- **La salida estructurada (`with_structured_output`) no es opcional para producción.** Es la única forma de garantizar un "contrato de datos" fiable entre el LLM y la aplicación que lo consume, eliminando la necesidad de parsing frágil.
- **RAG es el patrón esencial** para "aterrizar" (ground) los LLMs en conocimiento propietario.
- El prompting de un sistema RAG es crítico: la instrucción "Respondé SOLO con el contexto" es la principal defensa contra la **alucinación** cuando el conocimiento no existe en el corpus.
- La **observabilidad (`LangSmith`)** es fundamental para entender los trade-offs entre latencia (velocidad) y costo/calidad (tokens, complejidad de prompt) en cada paso de la cadena.

## Reflexión Personal
Esta práctica marcó la transición de la ingeniería de prompts (prompt engineering) al diseño de sistemas (systems design). El verdadero poder no reside solo en el LLM, sino en la **cadena de orquestación** que construimos a su alrededor. Se diseñó un sistema de extremo a extremo que combina recuperación de datos (RAG) con un formato de salida garantizado (Pydantic), creando una aplicación que es fiable, observable y soluciona una necesidad de negocio concreta (chatbot de soporte).
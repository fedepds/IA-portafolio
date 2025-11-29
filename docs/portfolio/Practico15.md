
-----

# Agente Cognitivo con LangGraph

## Contexto

En este proyecto implementé **LangGraph**, el framework estado del arte para construir agentes con lógica compleja, superando las limitaciones de cadenas lineales. Desarrollé un **Asistente de Soporte Académico** que demuestra:

- **Arquitectura ReAct (Reason + Act)**: Ciclos cognitivos donde el agente razona, actúa y observa antes de responder.
- **RAG avanzado**: Indexación de documentos institucionales (Reglamentos) en FAISS para fundamentar respuestas.
- **Tools integration**: Conexión del LLM con funciones deterministas (consultas a "bases de datos" simuladas).
- **StateGraph**: Diseño de flujos con nodos (Assistant, Tools, Memory) y aristas condicionales.
- **Interfaz Gradio**: Despliegue de UI interactiva para validación de usuario.

Este proyecto muestra cómo construir agentes conversacionales de nivel empresarial con memoria, herramientas y flujo dinámico.

---

## 🚀 Valor Agregado e Innovación

Este proyecto va más allá de tutoriales básicos de LangChain, demostrando arquitectura de sistemas complejos:

### 1. Arquitectura de Grafos Dirigidos con Lógica Condicional
- **No me limité a cadenas lineales (LangChain LCEL)**: Implementé un **StateGraph** con múltiples nodos y aristas condicionales que permiten flujos cíclicos.
- **Patrón ReAct implementado**: El agente NO responde inmediatamente, sino que:
  1. **Razona** (¿necesito herramientas?)
  2. **Actúa** (ejecuta tools: RAG, consulta DB)
  3. **Observa** (ve los resultados)
  4. **Decide** (bucle o respuesta final)
- **Valor**: Esta arquitectura permite agentes que pueden iterar, corregirse y tomar decisiones complejas, imposible con cadenas lineales simples.

### 2. Integración Dual: Memoria Semántica (RAG) + Herramientas Operativas
- **RAG para conocimiento no estructurado**: Indexación de documentos (Reglamento del curso) en FAISS para fundamentar respuestas sobre políticas/procedimientos.
- **Tools para datos estructurados**: Funciones Python simulando APIs de bases de datos (estado de alumnos, entregas pendientes).
- **Arquitectura híbrida**: El agente decide dinámicamente cuándo usar RAG vs tools vs ambos.
- **Valor**: Esta dualidad es crítica en sistemas reales donde hay conocimiento documental + datos transaccionales.

### 3. Gestión de Estado Persistente (Memory)
- **No es un chatbot sin memoria**: Implementé un nodo de `Memory` que resume la conversación y la incorpora al contexto del agente.
- **AgentState custom**: Diseñé una estructura de estado (`TypedDict`) con:
  - `messages`: historial conversacional
  - `summary`: resumen acumulativo (evita context overflow)
  - `user_id`: contexto del usuario actual
- **Valor**: Demostración de cómo gestionar estado en agentes de producción donde el contexto crece indefinidamente.

### 4. Debugging y Observabilidad de Grafos
- **Visualización del grafo**: Usé `graph.get_graph().draw_png()` para generar diagrama de flujo del agente (nodos + aristas condicionales).
- **Checkpoints implícitos**: LangGraph permite inspeccionar el estado en cada nodo, facilitando debugging.
- **Valor**: En sistemas complejos, la observabilidad es crítica; demostré cómo hacer agentes "inspeccionables".

### 5. Interfaz de Usuario con Gradio (Validación de UX)
- **No me quedé en el notebook**: Desplegué el agente en una interfaz web interactiva con Gradio.
- **Validación de caso de uso**: Permitió simular conversaciones reales y validar que:
  - El agente responde apropiadamente a preguntas sobre reglamentos (RAG)
  - El agente consulta correctamente datos de alumnos (tools)
  - El agente mantiene contexto entre turnos (memory)
- **Valor**: Demostración de thinking de producto ("cómo lo usaría un usuario real") vs solo implementación técnica.

### 6. Manejo de Casos Edge (Robustez)
- **Pregunta sin respuesta en RAG**: El sistema reportó "No encontré información sobre cómo cambiar la batería" (fallo controlado, no alucinación).
- **Validación de inputs**: Las tools validan que el `user_id` sea válido antes de consultar.
- **Instrucciones de System Prompt**: El LLM tiene instrucciones claras sobre cuándo usar tools (evita abuso de herramientas).
- **Valor**: Robustez y manejo de errores son características críticas para sistemas de producción.

---

## Objetivos

  * **Diseñar** una arquitectura de grafo cíclico (ReAct) utilizando `LangGraph` para orquestar la toma de decisiones del LLM.
  * [cite\_start]**Implementar** un sistema RAG (Retrieval-Augmented Generation) para fundamentar las respuestas en documentación institucional[cite: 1825].
  * **Integrar** herramientas deterministas (funciones Python) para simular consultas a bases de datos en tiempo real.
  * **Desplegar** una interfaz interactiva (Gradio) para validar la experiencia de usuario.

## Actividades (con tiempos estimados)

| Actividad | Tiempo | Resultado esperado |
| :--- | :--- | :--- |
| **Configuración de Entorno** | 30 min | Instalación de `langgraph`, `faiss-cpu`, y configuración de API Keys. |
| **Implementación RAG** | 45 min | Indexación de documentos del curso en Vector Store. |
| **Desarrollo de Tools** | 40 min | Creación de funciones para consulta de alumnos y binding con LLM. |
| **Construcción del Grafo** | 60 min | Definición de nodos (`Assistant`, `Tools`, `Memory`) y aristas condicionales. |
| **Interfaz de Usuario** | 30 min | Despliegue de chat interactivo con Gradio. |
| **Total Estimado** | **3h 25m** | **Agente funcional desplegado** |

## Desarrollo

### 1\. Arquitectura del Sistema

A diferencia de las cadenas lineales (`LangChain`), opté por una arquitectura basada en grafos (`LangGraph`) que permite bucles de retroalimentación.

\!\!\! note "Patrón ReAct (Reason + Act)"
El grafo implementa un ciclo cognitivo: **Pensamiento $\rightarrow$ Acción $\rightarrow$ Observación**. [cite\_start]Esto permite al modelo "ver" el resultado de una herramienta (ej. base de datos) antes de formular la respuesta final al usuario[cite: 1965].

### 2\. Herramientas y RAG

Implementé dos capacidades principales para el agente:

  * **Memoria Semántica (RAG):** Utilicé `FAISS` para indexar el reglamento del curso. Esto mitiga las alucinaciones al obligar al modelo a consultar la documentación oficial.
  * **Herramientas Operativas:** Funciones Python simulando una API de alumnos.

??? details "Ver código de definición de Tools"
\`\`\`python
@tool
def consultar\_reglamento(pregunta: str) -\> str:
"""Consulta sobre evaluación, fechas o contenido."""
docs = retriever.invoke(pregunta)
return "\\n".join([d.page\_content for d in docs])

````
@tool
def ver_estado_alumno(matricula: str) -> str:
    """Consulta estado académico en DB simulada."""
    # Lógica de consulta al diccionario DB_ALUMNOS
    ...
```
````

### 3\. Orquestación con LangGraph

La pieza central fue la definición del `StateGraph`. Utilicé un objeto de estado con `Annotated[list, operator.add]` para gestionar el historial de mensajes de forma acumulativa.

```python linenums="1"
# Definición del Router Condicional
def route_from_assistant(state: AgentState) -> str:
    last = state["messages"][-1]
    # Si el LLM pide usar una herramienta, desviamos el flujo
    if last.tool_calls:
        return "tools"
    return END

# Construcción del Grafo
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("memory", memory_node) # Nodo extra para resumen

# Ciclo de Aprendizaje: Tools -> Memory -> Assistant
builder.add_edge("tools", "memory")
builder.add_edge("memory", "assistant")
```

### 4\. Interfaz de Usuario

Para facilitar la validación por parte de stakeholders no técnicos, envolví el agente en una interfaz de chat usando `Gradio`.

## Evidencias

### Prueba de Integración

La siguiente prueba demuestra la capacidad del agente para combinar **RAG** (fecha de defensa) y **Datos Estructurados** (estado del alumno) en una sola respuesta coherente.

**Prompt del Usuario:**

> *"Soy el alumno A001. ¿Tengo entregas pendientes? Y recordame cuándo es la defensa final."*

**Log de Ejecución:**

```text
--- Paso del Agente ---
🔧 Tool invocada: ver_estado_alumno (Args: {'matricula': 'A001'})
--- Paso del Agente ---
🔧 Tool invocada: consultar_reglamento (Args: {'pregunta': 'fecha defensa final'})
--- Paso del Agente ---
🤖 Respuesta Final: Como alumno A001, no tienes entregas pendientes. La defensa final está programada para el 02/12.
```

## Reflexión

### Aprendizajes Clave 🧠

1.  **Estado vs. Stateless:** Comprendí que para conversaciones complejas, gestionar el estado explícitamente (`AgentState`) es superior a pasar cadenas de texto crudo.
2.  [cite\_start]**Orquestación:** LangGraph permite lógica condicional ("si pasa X, ve al nodo Y") que es imposible en cadenas secuenciales simples[cite: 1883].
3.  **Importancia del Prompting:** El modelo `gpt-4o-mini` necesita instrucciones claras en el *System Prompt* para no abusar de las herramientas.

### Exploraciones Futuras 🚀

  * **Persistencia:** Actualmente `FAISS` corre en memoria. [cite\_start]Para producción, podría migrarse a **Pinecone** o **Qdrant** para soportar millones de documentos y actualizaciones sin re-training[cite: 1853].
  * **Seguridad:** Implementar validación de inputs en las herramientas para evitar inyecciones o acceso a datos de otros alumnos.
  * **Privacidad:** El nodo de memoria resume la conversación. Podría agregarse un filtro para anonimizar datos personales (PII) antes de guardarlos en el resumen.

## Referencias

[cite\_start]\* [cite: 1825] **Generación Aumentada por Recuperación (RAG): Fundamentos.** *Investigación NLP y LLMs\_ Guía Detallada.pdf*.
[cite\_start]\* [cite: 1883] **Agentes Cíclicos con LangGraph.** *Investigación NLP y LLMs\_ Guía Detallada.pdf*.
[cite\_start]\* [cite: 1965] **Patrón ReAct (Reason + Act).** *Investigación NLP y LLMs\_ Guía Detallada.pdf*.
[cite\_start]\* [cite: 1853] **Almacenes Vectoriales (FAISS vs Pinecone).** *Investigación NLP y LLMs\_ Guía Detallada.pdf*.

  * Se utliza documentación oficial de [LangGraph](https://langchain-ai.github.io/langgraph/) y [Gradio](https://www.gradio.app/).

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT4/Practico15.ipynb)**
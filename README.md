# 🤖 MINIHAL9000 - Asistente IA con Voz y Memoria Semántica

Asistente de Telegram inteligente con procesamiento de voz, búsqueda semántica avanzada, ejecución de comandos y personalidad configurable. **Fase 3: Memoria SQLite con Embeddings**.

---

## ✨ Características Principales

### 🎤 Reconocimiento de Voz
- Transcripción automática de mensajes de voz en español
- Modelo Whisper (base) optimizado para CPU
- Comprensión de intenciones en lenguaje natural

### 🧠 Memoria Semántica Inteligente
- **Base de datos SQLite** con vectores de embeddings
- **Búsqueda semántica**: entiende sinónimos y variaciones
- Ejemplo: guarda "mi color favorito = azul" y luego pregunta "¿qué tonalidad me gusta?" → ¡lo encuentra!
- Historial con timestamps
- Gestión completa: guardar, buscar, listar, eliminar

### 📚 RAG (Retrieval Augmented Generation)
- Carga PDFs y los indexa automáticamente
- Búsqueda semántica en documentos
- Fragmentación inteligente con solapamiento
- Integración con respuestas de IA

### 🔧 Tool Calling Nativo
- Ollama con soporte para llamadas a funciones
- Ejecución segura de comandos Linux
- Lista blanca de comandos permitidos

### 🔊 Síntesis de Voz (TTS)
- Piper TTS en español
- Salida de audio natural
- Fragmentación automática de textos largos

### 🎭 Personalidad Configurable
- Sistema de roles y rasgos personalizables
- Almacenamiento en JSON persistente
- Cambio dinámico sin reiniciar

---

## 🚀 Instalación Rápida

### 1️⃣ Requisitos del Sistema

```bash
# CachyOS/Arch Linux
sudo pacman -S python-pip python ollama

# Debian/Ubuntu
sudo apt-get install python3-pip python3-venv ollama

# macOS (con Homebrew)
brew install python ollama
```

### 2️⃣ Configurar Ollama

```bash
# Iniciar el servicio
systemctl --user start ollama

# (Opcional) Habilitar en cada inicio
systemctl --user enable --now ollama

# Descargar modelo recomendado (10GB - 7B params)
ollama pull qwen2.5:7b-instruct-q4_K_M

# O para equipos con menos RAM (2GB - 1B params)
ollama pull granite4:1b
```

### 3️⃣ Entorno Virtual Python

```bash
cd /ruta/a/MINIHAL9000

# Crear entorno
python -m venv venv

# Activar
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows
```

### 4️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
python-telegram-bot==21.0.1
faster-whisper==1.0.3
chromadb==0.5.0
sentence-transformers==3.0.1
ollama==0.3.1
PyPDF2==4.2.1
```

### 5️⃣ Configurar Token de Telegram

Obtén tu token en [@BotFather](https://t.me/BotFather):

```python
# En asistente.py, línea ~18
TOKEN = "tu_token_aqui"
```

### 6️⃣ Configurar Rutas Locales

```python
# asistente.py - Rutas del sistema
PIPER_MODEL = "/ruta/a/tu/modelo/es_ES-davefx-medium.onnx"
RAG_PATH = os.path.expanduser("~/HAL9000/rag_db")
MEMORIA_DB = os.path.expanduser("~/.telegram_bot_memoria.db")
```

---

## 📖 Guía de Uso

### Comandos Básicos

#### 🎓 Enseñar Información (Memoria)
```
/aprender Mi nombre es Rorschach = Me llamas Rorscha a veces
/aprender Python es = Un lenguaje de programación poderoso
```

#### 🤔 Recordar Información
```
/recordar cómo me llamas
/recordar qué es Python
/recordar lenguaje de programación
```
✨ **La búsqueda entiende sinónimos y variaciones**

#### 📋 Ver Todo lo que Sabe
```
/lista
```

#### 🗑️ Borrar Información
```
/olvidar Mi nombre es Rorschach
```

#### 🎭 Cambiar Personalidad
```
/personalidad nombre = JARVIS
/personalidad rol = asistente de ciencia ficción
/personalidad rasgos = misterioso y poético
/personalidad idioma = inglés
```

#### 🔧 Ejecutar Comandos Linux
```
/comando ls -la
/comando df -h
/comando ps aux
/comando whoami
```

O **directamente en voz**:
```
🎤 "enséñame lo que hay en esta carpeta"
→ Ejecuta: ls -la

🎤 "cuánto espacio libre me queda"
→ Ejecuta: df -h

🎤 "qué programas están corriendo"
→ Ejecuta: ps aux
```

#### 💬 Conversación Natural
Solo escribe o envía un mensaje de voz y HAL9000 responderá inteligentemente.

---

## 📚 Gestión de PDFs y RAG

### Cargar un PDF

```bash
python cargar_pdf.py documento.pdf
```

**Qué hace:**
1. Extrae todo el texto del PDF
2. Lo fragmenta en chunks de 1000 caracteres
3. Genera embeddings para cada fragmento
4. Guarda en ChromaDB

**Salida esperada:**
```
📄 Procesando: documento.pdf
✂️ Generados 42 fragmentos
✅ Cargados 42 fragmentos a tu RAG
📊 Total documentos en RAG: 127
```

### Probar la RAG

```bash
python probar_rag.py
```

Hace una búsqueda de ejemplo para verificar que funciona.

### Cómo Usa HAL9000 la RAG

1. Cuando recibe un mensaje, busca en la RAG documentos relacionados
2. Añade el contexto al prompt del modelo IA
3. Genera respuestas mejoradas con información de tus documentos

Ejemplo:
```
Usuario: ¿Qué comando de Linux se usa para ver procesos?

[RAG busca en PDFs cargados...]
→ Encuentra: "ps aux - muestra todos los procesos en ejecución"

HAL9000 responde con contexto mejorado ✨
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────┐
│            TELEGRAM BOT (asistente.py)          │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ ENTRADA      │  │ MEMORIA      │            │
│  ├──────────────┤  ├──────────────┤            │
│  │ • Texto      │  │ • SQLite DB  │            │
│  │ • Voz        │  │ • Embeddings │            │
│  │   (Whisper)  │  │ • Búsqueda   │            │
│  │              │  │   semántica  │            │
│  └──────────────┘  └──────────────┘            │
│        │                    ▲                   │
│        ▼                    │                   │
│  ┌──────────────────────────────────────┐      │
│  │    PROCESAMIENTO (responder_hibrido) │      │
│  ├──────────────────────────────────────┤      │
│  │ 1. Buscar en RAG                     │      │
│  │ 2. Tool Calling (ejecutar comandos)  │      │
│  │ 3. Detección tradicional (fallback)  │      │
│  │ 4. IA (Ollama)                       │      │
│  └──────────────────────────────────────┘      │
│        ▼                    ▲                   │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ SALIDA       │  │ COMPONENTES  │            │
│  ├──────────────┤  ├──────────────┤            │
│  │ • Texto      │  │ • Ollama     │            │
│  │ • Voz        │  │ • ChromaDB   │            │
│  │   (Piper)    │  │ • Whisper    │            │
│  │              │  │ • Piper      │            │
│  └──────────────┘  └──────────────┘            │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🔐 Seguridad

### Comandos Permitidos
El bot solo permite ejecutar comandos de lectura segura:

```python
COMANDOS_PERMITIDOS = [
    "ls", "pwd", "date", "whoami", "echo", "cat", "df", "free", "ps",
    "uptime", "uname", "du", "find", "grep", "wc", "head", "tail", "tree", "ip"
]
```

### Validaciones
- ✅ Lista blanca de comandos
- ✅ Timeout de 30 segundos
- ✅ Validación de permisos
- ✅ Limitación de salida (1900 caracteres)

---

## 📊 Cómo Funciona la Memoria Semántica

### Proceso de Guardado
```python
Usuario: /aprender mi color favorito = azul

1. "mi color favorito" → Generar embedding (384 dimensiones)
2. Guardar en SQLite:
   - clave: "mi color favorito"
   - valor: "azul"
   - embedding: [0.123, -0.456, 0.789, ...]
   - fecha: "2026-05-17T14:30:00"

3. Crear índice para búsquedas rápidas
```

### Proceso de Búsqueda
```python
Usuario: /recordar qué tonalidad me gusta

1. "qué tonalidad me gusta" → Generar embedding
2. Calcular similitud coseno con TODOS los embeddings guardados
3. Retornar los resultados > 60% similitud
4. Mostrar el más similar (90% similitud → "mi color favorito = azul")
```

### Similitud Semántica
```
Pregunta original: "mi color favorito"
Embedding 1: [0.100, 0.200, 0.300, ...]

Preguntas que encuentra:
- "qué tonalidad me gusta" → 92% similitud ✅
- "mi tono preferido" → 88% similitud ✅
- "color que prefiero" → 85% similitud ✅
- "¿cómo te llamas?" → 15% similitud ❌
```

---

## 🛠️ Estructura de Archivos

```
MINIHAL9000/
├── asistente.py          # Bot principal (Fase 3)
├── cargar_pdf.py         # Utilidad para cargar PDFs
├── probar_rag.py         # Test de RAG
├── requirements.txt      # Dependencias Python
├── README.md             # Este archivo
├── ROADMAP               # Plan de desarrollo
└── rag_db/               # Base de datos ChromaDB (creado automáticamente)

~/.telegram_bot_memoria.db  # SQLite con memoria semántica
~/.telegram_bot_personalidad.json  # Configuración de personalidad
```

---

## 🔄 Flujo de Conversación Completo

### Ejemplo: Usuario envía mensaje de voz

```
1. 🎤 Usuario: "¿Cuál es mi color favorito?"
   │
   ├─ Whisper transcribe → "cuál es mi color favorito"
   │
   ├─ RAG busca en documentos → [no encontrado]
   │
   ├─ Tool Calling intenta → [no es comando]
   │
   ├─ 💾 Memoria semántica busca:
   │    └─ Encuentra "mi color favorito = azul" (92% similitud)
   │
   ├─ IA (Ollama) genera contexto mejorado
   │
   └─ 🔊 Piper TTS responde en voz: "Tu color favorito es azul"
```

---

## 🚨 Troubleshooting

### "No se pudo conectar a Ollama"
```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Si falla, iniciar manualmente
ollama serve
```

### "Error: Modelo no encontrado"
```bash
# Descargar el modelo
ollama pull qwen2.5:7b-instruct-q4_K_M

# Verificar disponible
ollama list
```

### "No se transcribe el audio"
```bash
# Verificar Whisper
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

# Si falla, descargar manualmente
```

### "Errores de permisos en Piper"
```bash
# Verificar ruta del modelo
ls -la /home/usuario/HAL9000/es_ES-davefx-medium.onnx

# Añadir permisos si es necesario
chmod +r /home/usuario/HAL9000/es_ES-davefx-medium.onnx
```

---

## 📈 Roadmap (Fases Futuras)

| Fase | Mejora | Estado |
|------|--------|--------|
| ✅ 1 | Mover reglas a archivo externo | Completado |
| ✅ 2 | Tool calling nativo + voz | Completado |
| ✅ 3 | Memoria SQLite con embeddings | ⭐ ACTUAL |
| 🔄 4 | Búsqueda híbrida (BM25 + embeddings) | Planeado |
| 🔄 5 | Análisis de imagen | Planeado |
| 🔄 6 | Contexto multi-usuario | Planeado |

---

## 💡 Ejemplos Avanzados

### 1. Base de Conocimiento Personal
```
/aprender Lenguaje favorito = Python
/aprender Framework web = Django
/aprender Base de datos = PostgreSQL

Usuario: "¿Con qué trabajo?"
→ HAL9000 responde inteligentemente usando memoria
```

### 2. Documentación Técnica
```bash
# Cargar tu documentación de proyectos
python cargar_pdf.py manual_proyecto.pdf
python cargar_pdf.py guia_arquitectura.pdf

Usuario: "¿Cuál es la arquitectura de mi sistema?"
→ Busca en RAG y responde con contexto de tus docs
```

### 3. Cambiar Personalidad Dinámicamente
```
/personalidad nombre = JARVIS
/personalidad rol = asistente de ingeniería
/personalidad rasgos = profesional, preciso, británico
/personalidad instrucciones_adicionales = Responde en inglés formal

Usuario: "hello, how are you?"
→ JARVIS responde como su carácter
```

---

## 📞 Soporte y Contribuciones

- 🐛 Reporta bugs en Issues
- 💭 Sugerencias en Discussions
- 🤝 PRs bienvenidas

---

## 📄 Licencia

Proyecto educativo. Úsalo libremente. 🚀

---

## 🎓 Conceptos Aprendidos

- **NLP con Embeddings**: Búsqueda semántica real
- **SQLite + JSON**: Persistencia eficiente
- **Tool Calling**: IA que ejecuta acciones
- **RAG**: Contexto aumentado en IA
- **TTS/STT**: Procesamiento de voz bidireccional
- **Telegram Bots**: API asíncrona moderna

---

**Versión Actual**: Fase 3 (Memoria Semántica) ⭐  
**Última Actualización**: 2026-05-17  
**Autor**: Bitriaferrero

---

*"The best time to plant a tree was 20 years ago. The second best time is now." - Hecho con ❤️ en CachyOS*

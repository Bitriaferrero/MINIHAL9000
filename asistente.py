#!/usr/bin/env python3
"""
Asistente de Telegram con Voz, RAG y Memoria Semántica.
FASE 3: Memoria con búsqueda semántica (embeddings)
"""

import subprocess
import tempfile
import os
import json
import re
import shlex
import time
import sqlite3
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from faster_whisper import WhisperModel
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama

# ========= CONFIGURACIÓN =========
TOKEN = "TOKENDETELEGRAM"

# Modelo de Ollama (con tool calling)
OLLAMA_MODEL = "granite4:1b"  # También funciona con "llama3.2:3b" o "mistral:7b"

# Ruta del modelo de voz Piper
PIPER_MODEL = "/home/rorshach/HAL9000/es_ES-davefx-medium.onnx"

# Archivos de memoria (SQLite para búsqueda semántica)
MEMORIA_DB = os.path.expanduser("~/.telegram_bot_memoria.db")
PERSONALIDAD_FILE = os.path.expanduser("~/.telegram_bot_personalidad.json")

# Ruta de la base de datos RAG
RAG_PATH = os.path.expanduser("~/HAL9000/rag_db")

# Comandos permitidos por seguridad
COMANDOS_PERMITIDOS = [
    "ls", "pwd", "date", "whoami", "echo", "cat", "df", "free", "ps", 
    "uptime", "uname", "du", "find", "grep", "wc", "head", "tail", "tree", "ip"
]

# ========= TOOL CALLING =========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ejecutar_comando",
            "description": "Ejecuta un comando en el sistema Linux",
            "parameters": {
                "type": "object",
                "properties": {
                    "comando": {
                        "type": "string",
                        "description": "El comando de Linux a ejecutar"
                    }
                },
                "required": ["comando"]
            }
        }
    }
]

# ========= INICIALIZAR MODELOS =========
print("🧠 Cargando modelo de embeddings...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modelo de embeddings listo")

# ========= MEMORIA SEMÁNTICA (SQLite + Embeddings) =========
def init_memoria_db():
    """Inicializa la base de datos SQLite para memoria semántica"""
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clave TEXT NOT NULL,
            valor TEXT NOT NULL,
            embedding BLOB NOT NULL,
            fecha TEXT NOT NULL
        )
    ''')
    # Crear índice para búsqueda rápida
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_clave ON memoria(clave)')
    conn.commit()
    conn.close()
    print(f"✅ Base de datos memoria: {MEMORIA_DB}")

def obtener_embedding(texto):
    """Genera embedding para un texto"""
    return embedder.encode(texto).tolist()

def guardar_en_memoria(clave, valor):
    """Guarda en memoria con embedding semántico"""
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    
    embedding = obtener_embedding(clave)
    embedding_blob = json.dumps(embedding)
    fecha = datetime.now().isoformat()
    
    # Verificar si ya existe
    cursor.execute('SELECT id FROM memoria WHERE clave = ?', (clave,))
    existe = cursor.fetchone()
    
    if existe:
        cursor.execute('''
            UPDATE memoria SET valor = ?, embedding = ?, fecha = ?
            WHERE clave = ?
        ''', (valor, embedding_blob, fecha, clave))
    else:
        cursor.execute('''
            INSERT INTO memoria (clave, valor, embedding, fecha)
            VALUES (?, ?, ?, ?)
        ''', (clave, valor, embedding_blob, fecha))
    
    conn.commit()
    conn.close()
    return True

def buscar_en_memoria(pregunta, umbral=0.6):
    """
    Busca en memoria por similitud semántica.
    Retorna: (clave, valor, similitud) o None
    """
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    
    # Obtener todas las memorias
    cursor.execute('SELECT clave, valor, embedding FROM memoria')
    resultados = cursor.fetchall()
    conn.close()
    
    if not resultados:
        return None
    
    # Generar embedding de la pregunta
    pregunta_embedding = obtener_embedding(pregunta)
    
    # Calcular similitud con cada memoria
    mejores = []
    for clave, valor, embedding_blob in resultados:
        embedding = json.loads(embedding_blob)
        
        # Calcular similitud coseno
        similitud = calcular_similitud(pregunta_embedding, embedding)
        
        if similitud > umbral:
            mejores.append((similitud, clave, valor))
    
    if not mejores:
        return None
    
    # Ordenar por similitud (mayor primero)
    mejores.sort(reverse=True)
    
    mejor = mejores[0]
    return {
        "clave": mejor[1],
        "valor": mejor[2],
        "similitud": mejor[0]
    }

def calcular_similitud(v1, v2):
    """Calcula similitud coseno entre dos vectores"""
    import math
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def listar_memoria_semantica():
    """Lista todas las claves guardadas"""
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT clave, fecha FROM memoria ORDER BY fecha DESC')
    resultados = cursor.fetchall()
    conn.close()
    return resultados

def eliminar_de_memoria(clave):
    """Elimina una entrada de memoria por clave exacta"""
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM memoria WHERE clave = ?', (clave,))
    afectadas = cursor.rowcount
    conn.commit()
    conn.close()
    return afectadas > 0

def limpiar_memoria_antigua(dias=30):
    """Elimina memorias más antiguas que X días"""
    conn = sqlite3.connect(MEMORIA_DB)
    cursor = conn.cursor()
    fecha_limite = (datetime.now().timestamp() - dias * 86400)
    cursor.execute('DELETE FROM memoria WHERE julianday(?) - julianday(fecha) > ?', 
                   (datetime.now().isoformat(), dias))
    afectadas = cursor.rowcount
    conn.commit()
    conn.close()
    return afectadas

# ========= DETECTOR TRADICIONAL (fallback) =========
PALABRAS_ACCION = {
    'listar': ['lista', 'listame', 'listar', 'muestra', 'muéstrame', 'enseña', 'enseñame', 'ver', 'mostrar', 'dame'],
    'consultar': ['cuánto', 'cuanta', 'dime', 'saber', 'consultar', 'preguntar', 'cual', 'qué', 'que', 'cómo', 'como']
}

OBJETOS_SISTEMA = {
    'archivos': ['archivos', 'ficheros', 'directorios', 'carpetas', 'contenido', 'carpeta', 'directorio'],
    'procesos': ['procesos', 'programas', 'aplicaciones', 'tareas', 'ejecutandose', 'corriendo'],
    'memoria': ['memoria', 'ram', 'libre', 'disponible', 'usada', 'ocupada'],
    'disco': ['disco', 'espacio', 'almacenamiento', 'gb', 'mb', 'duro', 'ssd'],
    'red': ['red', 'ip', 'conexión', 'wifi', 'internet', 'dirección'],
    'fecha': ['fecha', 'hora', 'día', 'tiempo', 'reloj'],
    'usuario': ['usuario', 'quien', 'nombre', 'cuenta'],
    'ubicacion': ['ubicación', 'donde', 'dónde', 'ruta', 'directorio actual']
}

MAPEO_COMANDOS = {
    ('archivos', 'listar'): 'ls -la',
    ('carpetas', 'listar'): 'ls -la',
    ('directorios', 'listar'): 'ls -la',
    ('procesos', 'listar'): 'ps aux',
    ('procesos', 'consultar'): 'ps aux | head -20',
    ('memoria', 'consultar'): 'free -h',
    ('memoria', 'listar'): 'free -h',
    ('disco', 'consultar'): 'df -h',
    ('disco', 'listar'): 'df -h',
    ('red', 'consultar'): 'ip addr',
    ('red', 'listar'): 'ip addr',
    ('fecha', 'consultar'): 'date',
    ('usuario', 'consultar'): 'whoami',
    ('ubicacion', 'consultar'): 'pwd',
}

def detectar_intencion_tradicional(texto):
    """Detección tradicional en español"""
    texto_lower = texto.lower()
    
    accion = None
    for accion_nombre, palabras in PALABRAS_ACCION.items():
        for palabra in palabras:
            if palabra in texto_lower:
                accion = accion_nombre
                break
        if accion:
            break
    
    if not accion:
        return False, None
    
    objeto = None
    for objeto_nombre, palabras in OBJETOS_SISTEMA.items():
        for palabra in palabras:
            if palabra in texto_lower:
                objeto = objeto_nombre
                break
        if objeto:
            break
    
    if not objeto:
        return False, None
    
    comando = MAPEO_COMANDOS.get((objeto, accion))
    if not comando:
        comando = MAPEO_COMANDOS.get((objeto, 'consultar'))
    
    if comando:
        return True, comando
    return False, None

# ========= INICIALIZAR RAG =========
print("📚 Cargando base de conocimiento RAG...")
try:
    chroma = PersistentClient(path=RAG_PATH)
    coleccion_rag = chroma.get_collection("conocimiento")
    print(f"✅ RAG cargada con {coleccion_rag.count()} fragmentos")
except Exception as e:
    print(f"⚠️ No se pudo cargar RAG: {e}")
    coleccion_rag = None

def buscar_en_rag(pregunta, max_resultados=5):
    if coleccion_rag is None:
        return None
    try:
        query_embedding = embedder.encode(pregunta).tolist()
        resultados = coleccion_rag.query(
            query_embeddings=[query_embedding],
            n_results=max_resultados
        )
        if resultados['documents'] and len(resultados['documents'][0]) > 0:
            contexto = ""
            for i, doc in enumerate(resultados['documents'][0], 1):
                fuente = "desconocida"
                if resultados['metadatas'][0][i-1] and isinstance(resultados['metadatas'][0][i-1], dict):
                    fuente = resultados['metadatas'][0][i-1].get('source', 'desconocida')
                contexto += f"[Fuente: {fuente}]\n{doc[:600]}\n\n"
            return contexto
        return None
    except Exception as e:
        print(f"Error en RAG: {e}")
        return None

# ========= PERSONALIDAD =========
PERSONALIDAD_DEFECTO = {
    "nombre": "HAL9000",
    "rol": "asistente personal experto en Linux",
    "rasgos": "inteligente, preciso, útil y amigable",
    "idioma": "español",
    "instrucciones_adicionales": """Responde de forma clara y concisa. Si no sabes algo, dilo honestamente.
Si el usuario pide ejecutar una acción, usa la herramienta disponible."""
}

def cargar_personalidad():
    if os.path.exists(PERSONALIDAD_FILE):
        with open(PERSONALIDAD_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return PERSONALIDAD_DEFECTO.copy()

def guardar_personalidad(personalidad):
    with open(PERSONALIDAD_FILE, 'w', encoding='utf-8') as f:
        json.dump(personalidad, f, ensure_ascii=False, indent=2)

def construir_system_prompt():
    p = cargar_personalidad()
    return f"""Eres {p['nombre']}, un {p['rol']} con personalidad {p['rasgos']}.
Hablas en {p['idioma']}.
{p['instrucciones_adicionales']}"""

# ========= WHISPER =========
print("🎤 Cargando modelo de voz (Whisper base - español)...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Modelo de voz listo")

def transcribir_audio(audio_path):
    try:
        segments, _ = whisper_model.transcribe(audio_path, language="es", beam_size=5)
        texto = " ".join([seg.text for seg in segments])
        return texto.strip() if texto else ""
    except Exception as e:
        print(f"Error transcripción: {e}")
        return ""

# ========= PIPER TTS =========
def hablar_frase(texto):
    if not texto or len(texto.strip()) == 0:
        return False
    
    texto = texto.strip()
    
    # Eliminar emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    texto = emoji_pattern.sub('', texto)
    
    # Reemplazar caracteres problemáticos
    texto = texto.replace('`', "'")
    texto = texto.replace('"', ' ')
    texto = texto.replace('\\', ' ')
    texto = texto.replace('$', ' ')
    texto = texto.replace('\n', ' ')
    texto = texto.replace('¡', '')
    texto = texto.replace('¿', '')
    texto = texto.replace('*', ' ')
    texto = texto.replace('_', ' ')
    texto = texto.replace('[', ' ')
    texto = texto.replace(']', ' ')
    texto = texto.replace('(', ' ')
    texto = texto.replace(')', ' ')
    texto = texto.replace('{', ' ')
    texto = texto.replace('}', ' ')
    
    if len(texto) > 250:
        texto = texto[:247] + "..."
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name
        
        texto_escapado = shlex.quote(texto)
        cmd = f'echo {texto_escapado} | piper --model {PIPER_MODEL} --output_file {output_file}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error Piper: {result.stderr}")
            return False
        
        subprocess.run(f'aplay {output_file}', shell=True, check=True, capture_output=True, timeout=10)
        os.unlink(output_file)
        return True
        
    except subprocess.TimeoutExpired:
        print("Error TTS: Timeout en frase")
        return False
    except Exception as e:
        print(f"Error TTS: {e}")
        return False

def hablar(texto):
    if not texto or len(texto.strip()) == 0:
        return False
    
    frases = re.split(r'(?<=[.!?;])\s+', texto)
    
    if len(frases) == 1 and len(texto) < 300:
        return hablar_frase(texto)
    
    exito = True
    for i, frase in enumerate(frases):
        frase = frase.strip()
        if frase and len(frase) > 5:
            print(f"🔊 Hablando frase {i+1}/{len(frases)}: {frase[:50]}...")
            if not hablar_frase(frase):
                exito = False
            time.sleep(0.15)
    
    return exito

# ========= EJECUTAR COMANDO =========
def ejecutar_comando(comando):
    cmd_parts = comando.strip().split()
    if not cmd_parts or cmd_parts[0] not in COMANDOS_PERMITIDOS:
        return f"❌ Comando no permitido.\nPermitidos: {', '.join(COMANDOS_PERMITIDOS)}"
    try:
        resultado = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
        salida = resultado.stdout or resultado.stderr or "✅ Ejecutado sin salida."
        if len(salida) > 1900:
            salida = salida[:1900] + "\n... (truncado)"
        return f"```\n{salida}\n```"
    except subprocess.TimeoutExpired:
        return "⚠️ El comando tardó demasiado (30s límite)"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ========= IA HÍBRIDA =========
def responder_hibrido(mensaje, contexto_rag=None):
    system_prompt = construir_system_prompt()
    if contexto_rag:
        system_prompt += f"\n\n**Información de mi base de conocimiento:**\n{contexto_rag}"
    
    # Tool calling
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': mensaje}
            ],
            tools=TOOLS
        )
        
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tool in response.message.tool_calls:
                if tool.function.name == "ejecutar_comando":
                    comando = tool.function.arguments.get("comando", "")
                    if comando and comando.split()[0] in COMANDOS_PERMITIDOS:
                        print(f"🔧 Tool Calling ejecutó: {comando}")
                        return ejecutar_comando(comando)
    except Exception as e:
        print(f"⚠️ Tool Calling falló: {e}")
    
    # Detección tradicional
    es_comando, comando = detectar_intencion_tradicional(mensaje)
    if es_comando and comando:
        print(f"🔧 Detección tradicional ejecutó: {comando}")
        return ejecutar_comando(comando)
    
    # Respuesta normal
    try:
        respuesta = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': mensaje}
            ]
        )
        return respuesta['message']['content']
    except Exception as e:
        return f"❌ Error con IA: {e}"

# ========= MANEJADORES DE TELEGRAM =========
async def start(update, context):
    personalidad = cargar_personalidad()
    await update.message.reply_text(
        f"🤖 **{personalidad['nombre']}** (Fase 3 - Memoria Semántica)\n\n"
        f"Soy tu {personalidad['rol']}.\n\n"
        "**🎤 Por voz entiendo:**\n"
        "• 'enséñame lo que hay en esta carpeta' → `ls`\n"
        "• 'quiero ver cuánto espacio libre me queda' → `df -h`\n"
        "• 'dime qué programas están funcionando' → `ps aux`\n\n"
        "**📝 Memoria semántica (nuevo):**\n"
        "• `/aprender clave = valor` - Guarda información\n"
        "• `/recordar texto` - Busca por SIGNIFICADO (entiende sinónimos)\n"
        "• `/lista` - Ver todo lo que sé\n"
        "• `/olvidar clave` - Borrar información\n\n"
        "**Ejemplo de memoria semántica:**\n"
        "`/aprender mi color favorito = azul`\n"
        " luego preguntas: *'¿qué tonalidad me gusta?'* → ¡Encuentra 'azul'!\n\n"
        "**Otros comandos:**\n"
        "/comando cmd - Forzar ejecución\n"
        "/personalidad campo = valor - Cambiar mi personalidad",
        parse_mode='Markdown'
    )

async def cmd_aprender(update, context):
    texto = " ".join(context.args)
    if not texto or "=" not in texto:
        await update.message.reply_text(
            "❌ Uso: `/aprender clave = valor`\n\n"
            "Ejemplo: `/aprender mi color favorito = azul`\n"
            "Luego puedes preguntar '¿qué tonalidad me gusta?' y lo recordará.",
            parse_mode='Markdown'
        )
        return
    clave, valor = texto.split("=", 1)
    clave, valor = clave.strip(), valor.strip()
    guardar_en_memoria(clave, valor)
    await update.message.reply_text(
        f"✅ Aprendido:\n**{clave}** = `{valor}`\n\n"
        f"💡 Podrás preguntarlo de muchas formas diferentes gracias a la búsqueda semántica.",
        parse_mode='Markdown'
    )

async def cmd_recordar(update, context):
    if not context.args:
        await update.message.reply_text(
            "❌ Uso: `/recordar texto`\n\n"
            "Ejemplo: `/recordar color favorito`\n"
            "Entiende preguntas aunque no uses las palabras exactas.",
            parse_mode='Markdown'
        )
        return
    pregunta = " ".join(context.args).strip()
    
    resultado = buscar_en_memoria(pregunta)
    
    if resultado:
        similitud = resultado['similitud'] * 100
        await update.message.reply_text(
            f"🧠 **Recordatorio encontrado** (similitud: {similitud:.1f}%)\n\n"
            f"📌 **Pregunta original:** {resultado['clave']}\n"
            f"💬 **Respuesta:** {resultado['valor']}",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"❌ No recuerdo nada relacionado con *\"{pregunta}\"*\n\n"
            f"💡 Puedes usar `/aprender clave = valor` para enseñarme.",
            parse_mode='Markdown'
        )

async def cmd_lista(update, context):
    resultados = listar_memoria_semantica()
    if not resultados:
        await update.message.reply_text(
            "📭 No tengo nada en memoria.\n"
            "Usa `/aprender clave = valor` para enseñarme.",
            parse_mode='Markdown'
        )
        return
    
    lista = "\n".join([f"• `{clave}` (desde {fecha[:10]})" for clave, fecha in resultados[:20]])
    total = len(resultados)
    mensaje = f"📚 **Lo que sé ({total} items):**\n\n{lista}"
    if total > 20:
        mensaje += f"\n\n... y {total - 20} más."
    await update.message.reply_text(mensaje, parse_mode='Markdown')

async def cmd_olvidar(update, context):
    if not context.args:
        await update.message.reply_text("❌ Uso: `/olvidar clave`", parse_mode='Markdown')
        return
    clave = " ".join(context.args).strip()
    if eliminar_de_memoria(clave):
        await update.message.reply_text(f"🗑️ Olvidado: **{clave}**", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ No encontré **{clave}**", parse_mode='Markdown')

async def cmd_personalidad(update, context):
    texto = " ".join(context.args)
    if not texto or "=" not in texto:
        await update.message.reply_text(
            "❌ Uso: `/personalidad campo = valor`\n\n"
            "Campos: `nombre`, `rol`, `rasgos`, `idioma`, `instrucciones_adicionales`",
            parse_mode='Markdown'
        )
        return
    campo, valor = texto.split("=", 1)
    campo, valor = campo.strip(), valor.strip()
    personalidad = cargar_personalidad()
    if campo in personalidad:
        personalidad[campo] = valor
        guardar_personalidad(personalidad)
        await update.message.reply_text(f"✅ **{campo}** cambiado a: `{valor}`", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ Campo '{campo}' no válido.")

async def cmd_comando(update, context):
    comando = " ".join(context.args)
    if not comando:
        await update.message.reply_text("❌ Uso: `/comando ls -la`", parse_mode='Markdown')
        return
    resultado = ejecutar_comando(comando)
    await update.message.reply_text(f"📟 `{comando}`\n{resultado}", parse_mode='Markdown')
    hablar(resultado)

async def manejar_voz(update, context):
    await update.message.reply_text("🎤 Escuchando...")
    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            audio_path = tmp.name
        
        texto = transcribir_audio(audio_path)
        os.unlink(audio_path)
        
        if not texto:
            await update.message.reply_text("❌ No entendí el audio. ¿Puedes repetir?")
            return
        
        await update.message.reply_text(f"📝 Entendí: *\"{texto}\"*", parse_mode='Markdown')
        
        # Buscar en RAG
        contexto_rag = buscar_en_rag(texto)
        
        # Responder
        respuesta = responder_hibrido(texto, contexto_rag)
        
        await update.message.reply_text(respuesta)
        hablar(respuesta)
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def manejar_texto(update, context):
    texto = update.message.text
    
    # Comando directo
    primer_palabra = texto.split()[0] if texto.split() else ""
    if primer_palabra in COMANDOS_PERMITIDOS:
        resultado = ejecutar_comando(texto)
        await update.message.reply_text(f"📟 `{texto}`\n{resultado}", parse_mode='Markdown')
        hablar(resultado)
        return
    
    # Buscar en RAG
    contexto_rag = buscar_en_rag(texto)
    
    # Responder
    respuesta = responder_hibrido(texto, contexto_rag)
    
    await update.message.reply_text(respuesta)
    hablar(respuesta)

# ========= MAIN =========
def main():
    # Inicializar base de datos de memoria semántica
    init_memoria_db()
    
    personalidad = cargar_personalidad()
    print("=" * 50)
    print("🤖 ASISTENTE INTELIGENTE - FASE 3 (MEMORIA SEMÁNTICA)")
    print("=" * 50)
    print(f"📦 Modelo Ollama: {OLLAMA_MODEL}")
    print(f"🎤 Whisper: base (español)")
    print(f"🔊 Piper TTS: {PIPER_MODEL}")
    print(f"🎭 Personalidad: {personalidad['nombre']}")
    print(f"📚 RAG: {RAG_PATH}")
    if coleccion_rag:
        print(f"📊 Fragmentos en RAG: {coleccion_rag.count()}")
    print(f"💾 Memoria semántica: {MEMORIA_DB}")
    print("=" * 50)
    print("✅ Bot listo para usar")
    print("💡 La memoria ahora entiende sinónimos y lenguaje natural")
    print("=" * 50)
    
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("aprender", cmd_aprender))
    app.add_handler(CommandHandler("recordar", cmd_recordar))
    app.add_handler(CommandHandler("lista", cmd_lista))
    app.add_handler(CommandHandler("olvidar", cmd_olvidar))
    app.add_handler(CommandHandler("personalidad", cmd_personalidad))
    app.add_handler(CommandHandler("comando", cmd_comando))
    app.add_handler(MessageHandler(filters.VOICE, manejar_voz))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_texto))
    
    app.run_polling()

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Cargar RAG
chroma = PersistentClient(path='./rag_db')
coleccion = chroma.get_collection("conocimiento")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pregunta de prueba
pregunta = "¿Qué comando de Linux se usa para ver los procesos en ejecución?"

# Generar embedding
query_embedding = embedder.encode(pregunta).tolist()

# Buscar
resultados = coleccion.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(f"\n📝 Pregunta: {pregunta}\n")
print(f"🔍 Resultados encontrados: {len(resultados['documents'][0])}\n")

for i, doc in enumerate(resultados['documents'][0], 1):
    # Manejar metadatos que pueden ser None
    metadatos = resultados['metadatas'][0][i-1]
    if metadatos and isinstance(metadatos, dict):
        fuente = metadatos.get('source', 'desconocida')
    else:
        fuente = 'desconocida'
    
    print(f"--- Fuente {i}: {fuente} ---")
    print(doc[:500])
    print("...\n")

#!/usr/bin/env python3
import sys, hashlib, os
from PyPDF2 import PdfReader
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("📚 Uso: python cargar_pdf.py archivo.pdf")
    sys.exit(1)

if not os.path.exists(sys.argv[1]):
    print(f"❌ No encuentro el archivo: {sys.argv[1]}")
    sys.exit(1)

print(f"📄 Procesando: {sys.argv[1]}")

chroma = PersistentClient(path='./rag_db')
coleccion = chroma.get_or_create_collection("conocimiento")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Leer PDF
reader = PdfReader(sys.argv[1])
texto_completo = ""
for i, page in enumerate(reader.pages):
    texto = page.extract_text()
    if texto:
        texto_completo += f"\n--- Página {i+1} ---\n{texto}"

if not texto_completo.strip():
    print("❌ No se pudo extraer texto del PDF (¿puede ser escaneado?)")
    sys.exit(1)

# Fragmentar
chunk_size = 1000
overlap = 200
fragmentos = []
i = 0
while i < len(texto_completo):
    chunk = texto_completo[i:i + chunk_size]
    if chunk.strip():
        fragmentos.append(chunk)
    i += chunk_size - overlap

print(f"✂️ Generados {len(fragmentos)} fragmentos")

# Guardar
for idx, frag in enumerate(fragmentos):
    embedding = embedder.encode(frag[:1000]).tolist()  # Limitar por si acaso
    doc_id = hashlib.md5(f"{sys.argv[1]}_{idx}".encode()).hexdigest()
    coleccion.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[frag],
        metadatas=[{"source": os.path.basename(sys.argv[1]), "chunk": idx}]
    )
    if (idx + 1) % 100 == 0:
        print(f"   Procesados {idx + 1}/{len(fragmentos)}...")

print(f"✅ Cargados {len(fragmentos)} fragmentos a tu RAG")
print(f"📊 Total documentos en RAG: {coleccion.count()}")

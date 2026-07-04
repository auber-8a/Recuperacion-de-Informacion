import json
import os

# Ruta del notebook
path = "07vectordb.ipynb"

if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f: 
        data = json.load(f)

    # Eliminamos el metadato conflictivo de los widgets manteniendo los outputs
    if "widgets" in data.get("metadata", {}):
        del data["metadata"]["widgets"]
        print("✨ ¡Metadatos conflictivos eliminados! Las salidas se mantuvieron intactas.")
    else:
        print("✅ El archivo no tenía metadatos conflictivos en 'metadata.widgets'.")

    with open(path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2)
else:
    print(f"❌ No se encontró el archivo en la ruta: {path}")
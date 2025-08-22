# src/data/download_gtzan.py
import os
import zipfile
from pathlib import Path
import subprocess
import sys

DATA_DIR = Path("data/raw/gtzan")
DATA_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre"  # Kaggle community mirror

def have_kaggle_cli() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def download_with_kaggle():
    print("-> Descargando GTZAN desde Kaggle…")
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DATA_DIR)]
    subprocess.run(cmd, check=True)
    # Encuentra el zip descargado
    zips = list(DATA_DIR.glob("*.zip"))
    if not zips:
        raise RuntimeError("No se encontró el ZIP descargado de Kaggle.")
    zip_path = zips[0]
    print(f"-> Descomprimiendo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)
    print("-> Descarga y extracción completas.")

def main():
    if not have_kaggle_cli():
        print("ERROR: No se encontró la CLI de Kaggle. Instálala con: pip install kaggle")
        print("Y configura tu kaggle.json en ~/.kaggle o C:\\Users\\<usuario>\\.kaggle")
        sys.exit(1)
    download_with_kaggle()
    # Comprobación simple
    genres_dir = DATA_DIR / "Data" / "genres_original"
    if not genres_dir.exists():
        # Algunos mirrors usan 'genres' o 'genres_original'
        alt = DATA_DIR / "genres_original"
        if alt.exists():
            genres_dir = alt
    assert genres_dir.exists(), f"No se encontró carpeta de géneros en {genres_dir}"
    print(f"✅ Dataset listo en: {genres_dir.resolve()}")

if __name__ == "__main__":
    main()

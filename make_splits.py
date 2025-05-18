import os
import zipfile
import requests
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split


datasets = [
    'https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/kk-ru.txt.zip',
    'https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/kk-ru.txt.zip'
]

def download_and_extract(url, out_dir):
    fname = url.split("/")[-1]
    zip_path = out_dir / fname
    extract_path = out_dir / fname.replace(".zip", "")

    if not zip_path.exists():
        print(f"Downloading {fname}...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path

def load_pairs(path):
    kk_files = sorted(path.glob("*.kk"))
    ru_files = sorted(path.glob("*.ru"))
    pairs = []

    for kk_file, ru_file in zip(kk_files, ru_files):
        with open(kk_file, encoding="utf-8") as f_kk, open(ru_file, encoding="utf-8") as f_ru:
            kk_lines = [line.strip() for line in f_kk]
            ru_lines = [line.strip() for line in f_ru]
            for k, r in zip(kk_lines, ru_lines):
                if k and r:
                    pairs.append((k, r))
    return pairs

def compute_similarities(pairs):
    scores = []
    for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
        batch = pairs[i:i + BATCH_SIZE]
        kk_batch = [p[0] for p in batch]
        ru_batch = [p[1] for p in batch]
        emb_kk = model.encode(kk_batch, convert_to_tensor=True)
        emb_ru = model.encode(ru_batch, convert_to_tensor=True)
        sim = util.cos_sim(emb_kk, emb_ru).diag().cpu().numpy()
        scores.extend(sim)
    return np.array(scores)



# -------------------- CONFIG --------------------
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("cleaned_data")
SAVE_DIR.mkdir(exist_ok=True)
model = SentenceTransformer("sentence-transformers/LaBSE").to(DEVICE)


if __name__ == "__main__":
    all_pairs = []

    for url in datasets:
        extracted = download_and_extract(url, SAVE_DIR)
        pairs = load_pairs(extracted)
        all_pairs.extend(pairs)

    print(f"Total sentence pairs loaded: {len(all_pairs)}")
    similarities = compute_similarities(all_pairs)

    with open(SAVE_DIR / "all_pairs.txt", "w", encoding="utf-8") as f:
        for (k, r), s in zip(all_pairs, similarities):
            f.write(f"{k}\t{r}\n")

    np.savetxt(SAVE_DIR / "similarities.txt", similarities, fmt="%.6f")
    
    df = pd.DataFrame({
        "idx": np.arange(len(all_pairs)),
        "kazakh": [k for k, _ in all_pairs],
        "russian": [r for _, r in all_pairs],
        "score": similarities
    })
    df.to_csv(SAVE_DIR / "filter.tsv", sep="\t", index=False)

    # Split into train/test (80/20)
    train_idx, test_idx = train_test_split(np.arange(len(all_pairs)), test_size=0.2, random_state=42)

    with open(SAVE_DIR / "train.txt", "w", encoding="utf-8") as f_train, \
        open(SAVE_DIR / "test.txt", "w", encoding="utf-8") as f_test:
        for i in train_idx:
            f_train.write(f"{all_pairs[i][0]}\t{all_pairs[i][1]}\n")
        for i in test_idx:
            f_test.write(f"{all_pairs[i][0]}\t{all_pairs[i][1]}\n")

    print("✅ Done. Files saved in:", SAVE_DIR.absolute())

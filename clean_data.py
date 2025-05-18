import os
import zipfile
import requests
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ========== CONFIG ==========
BATCH_SIZE = 64
MAX_PAIRS = None
KK_FILE = "MultiCCAligned.kk-ru.kk"
RU_FILE = "MultiCCAligned.kk-ru.ru"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# 1. Download and unzip if needed
def download_and_extract():
    url = "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/kk-ru.txt.zip"
    zip_file = "kk-ru.txt.zip"

    if not os.path.exists(zip_file):
        print("Downloading dataset...")
        r = requests.get(url)
        with open(zip_file, "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

# 2. Load aligned sentences
def load_data():
    with open(KK_FILE, encoding="utf-8") as f_kk, \
         open(RU_FILE, encoding="utf-8") as f_ru:
        kk = [line.strip() for line in f_kk]
        ru = [line.strip() for line in f_ru]
    return kk, ru

# 3. Compute similarity scores
def compute_similarity_array(kk_sentences, ru_sentences):
    model = SentenceTransformer('sentence-transformers/LaBSE').to(DEVICE)

    all_scores = []

    for i in tqdm(range(0, len(kk_sentences), BATCH_SIZE)):
        batch_kk = kk_sentences[i:i + BATCH_SIZE]
        batch_ru = ru_sentences[i:i + BATCH_SIZE]

        emb_kk = model.encode(batch_kk, convert_to_tensor=True, batch_size=32)
        emb_ru = model.encode(batch_ru, convert_to_tensor=True, batch_size=32)

        scores = util.cos_sim(emb_kk, emb_ru).diag().cpu().numpy()
        all_scores.extend(scores)

    return np.array(all_scores)



# ========== RUN ==========
if __name__ == "__main__":
    download_and_extract()
    kk, ru = load_data()
    if MAX_PAIRS:
        kk = kk[:MAX_PAIRS]
        ru = ru[:MAX_PAIRS]
    print(f"Loaded {len(kk)} sentence pairs.")

    score_array = compute_similarity_array(kk, ru)
    print("Done. Example scores:", score_array[:10])

    # (Optional) Save if needed
    np.save("ru_kk_similarity_scores.npy", score_array)
    np.savetxt("ru_kk_similarity_scores.txt", score_array, fmt="%.6f")
    print(len(score_array <= 0.65))


import os
import re
import json
import random
import requests

# 23 splits in IndicCorpV2
SPLIT_TO_FILE = {
    "asm_Beng": "data/as.txt",
    "ben_Beng": "data/bn.txt",
    "brx_Deva": "data/bd.txt",
    "doi_Deva": "data/dg.txt",
    "gom_Deva": "data/gom.txt",
    "guj_Gujr": "data/gu.txt",
    "hin_Deva": "data/hi-1.txt",
    "kan_Knda": "data/kn.txt",
    "kas_Arab": "data/ks.txt",
    "mai_Deva": "data/mai.txt",
    "mal_Mlym": "data/ml.txt",
    "mar_Deva": "data/mr.txt",
    "mni_Mtei": "data/mni.txt",
    "npi_Deva": "data/ne.txt",
    "ory_Orya": "data/or.txt",
    "pan_Guru": "data/pa.txt",
    "san_Deva": "data/sa.txt",
    "snd_Deva": "data/sd.txt",
    "tam_Taml": "data/ta.txt",
    "tel_Telu": "data/te.txt",
    "urd_Arab": "data/ur.txt",
    "khasi": "data/kha.txt",
    "santhali": "data/sat.txt"
}

BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/"

def tokenize_paragraph_to_sentences(paragraph):
    """
    Tokenizes a paragraph into sentences using standard Indic and Latin sentence boundary delimiters.
    Delimiters: purna viram (। \u0964), double purna viram (॥ \u0965), period (.), exclamation (!), question mark (?)
    """
    # Pattern to split by delimiters while keeping non-empty results
    raw_sentences = re.split(r'[।॥.!?\n]+', paragraph)
    sentences = []
    for s in raw_sentences:
        cleaned = s.strip()
        # Filter sentences to ensure clean, meaningful text (minimum 15 characters)
        if len(cleaned) >= 15:
            sentences.append(cleaned)
    return sentences

def fetch_sentences_for_language(lang_code, relative_path, target_count=1200):
    url = BASE_URL + relative_path
    chunk_size = 500000  # 500 KB chunk
    start_byte = 0
    sentences = []
    
    print(f"Fetching sentences for split '{lang_code}'...")
    
    while len(sentences) < target_count and start_byte < 15000000:  # limit to ~15MB max fetch
        headers = {'Range': f'bytes={start_byte}-{start_byte + chunk_size - 1}'}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code not in (200, 206):
                print(f"Warning: Received status code {resp.status_code} for {lang_code}")
                break
            
            text_chunk = resp.content.decode('utf-8', errors='ignore')
            if not text_chunk:
                break
                
            paragraphs = text_chunk.split('\n')
            for p in paragraphs:
                p_sentences = tokenize_paragraph_to_sentences(p)
                for s in p_sentences:
                    sentences.append(s)
                    if len(sentences) >= target_count:
                        break
                if len(sentences) >= target_count:
                    break
                    
            start_byte += chunk_size
        except Exception as e:
            print(f"Error fetching chunk for {lang_code}: {e}")
            break
            
    print(f"  -> Collected {len(sentences)} sentences for '{lang_code}'")
    return sentences[:target_count]

def build_dataset(output_path="indiccorp_dataset.json", seed=42):
    random.seed(seed)
    train_data = []
    val_data = []
    test_data = []
    
    language_labels = list(SPLIT_TO_FILE.keys())
    
    print("=== Building Dataset from IndicCorpV2 (1,200 samples/lang) ===")
    for lang_code, rel_path in SPLIT_TO_FILE.items():
        sentences = fetch_sentences_for_language(lang_code, rel_path, target_count=1200)
        
        if len(sentences) < 1200:
            print(f"Warning: Only collected {len(sentences)} sentences for {lang_code}")
            
        # Shuffle for randomness
        random.shuffle(sentences)
        
        # Stratified Split 80:10:10 (960 train, 120 val, 120 test)
        train_samples = sentences[:960]
        val_samples = sentences[960:1080]
        test_samples = sentences[1080:1200]
        
        for s in train_samples:
            train_data.append({"text": s, "label": lang_code})
        for s in val_samples:
            val_data.append({"text": s, "label": lang_code})
        for s in test_samples:
            test_data.append({"text": s, "label": lang_code})
            
    dataset = {
        "labels": language_labels,
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\nDataset construction complete!")
    print(f"Total Train samples: {len(train_data)} ({len(train_data)//len(language_labels)} per lang)")
    print(f"Total Val samples:   {len(val_data)} ({len(val_data)//len(language_labels)} per lang)")
    print(f"Total Test samples:  {len(test_data)} ({len(test_data)//len(language_labels)} per lang)")
    print(f"Saved dataset to '{output_path}'")
    return dataset

if __name__ == "__main__":
    build_dataset()

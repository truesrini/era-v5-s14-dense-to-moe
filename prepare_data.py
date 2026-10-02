"""
Download TinyStories (one training shard + validation), train a small byte-level BPE
tokenizer on it, and write the token ids to data/train.bin and data/val.bin (uint16).

    python prepare_data.py
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "roneneldan/TinyStories"
OUT = "data"
VOCAB = 8192
TRAIN_SHARDS = 1          # one shard (~530k stories) is far more than we train on
EOS = "<|endoftext|>"


def load_texts(files):
    texts = []
    for f in files:
        path = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=os.path.join(OUT, "hf"))
        texts += [t.strip() for t in pq.read_table(path).column("text").to_pylist() if t and t.strip()]
    return texts


def encode_to_bin(tok, texts, path, eos_id):
    ids = []
    for i in range(0, len(texts), 10_000):
        for enc in tok.encode_batch(texts[i:i + 10_000]):
            ids.extend(enc.ids)
            ids.append(eos_id)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    train_files = [f for f in files if "train" in f][:TRAIN_SHARDS]
    val_files = [f for f in files if "validation" in f]
    print("train shards:", train_files, "\nval shards:", val_files)

    train_texts, val_texts = load_texts(train_files), load_texts(val_files)
    print(f"stories: train {len(train_texts):,}  val {len(val_texts):,}")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_texts[:200_000], trainer=trainer)
    tok.save(os.path.join(OUT, "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)

    n_train = encode_to_bin(tok, train_texts, os.path.join(OUT, "train.bin"), eos_id)
    n_val = encode_to_bin(tok, val_texts, os.path.join(OUT, "val.bin"), eos_id)
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id,
            "train_tokens": n_train, "val_tokens": n_val,
            "train_files": train_files, "val_files": val_files}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

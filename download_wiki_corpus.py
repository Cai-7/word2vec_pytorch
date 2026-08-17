"""Download preprocessed Wikipedia text corpus for word2vec training.

Modes:
  text8  —   100M chars, 71K vocab, ~100MB (default)
  text9  —  1000M chars, ~200K vocab, ~1GB (recommended for analogy eval)
  full   — Full Wikipedia via HuggingFace datasets, ~20GB

Usage:
    python download_wiki_corpus.py                     # text8
    python download_wiki_corpus.py --mode text9         # text9 (recommended)
    python download_wiki_corpus.py --mode full          # full Wikipedia

Output:
    wiki_train.txt — one sentence per line, lowercase, ready for word2vec.py
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile
from io import BytesIO


BASE_URL = "http://mattmahoney.net/dc/"
FALLBACK_URL = "https://github.com/ranihorev/text/raw/master/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia corpus for word2vec training"
    )
    parser.add_argument(
        "--mode", default="text9",
        help='"text8" (100M chars), "text9" (1B chars, default, recommended), "full" (all Wikipedia)'
    )
    parser.add_argument(
        "--output", default="wiki_train.txt",
        help="Output file path (default: wiki_train.txt)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=80,
        help="Words per line when processing text8/text9 (default: 80)"
    )
    return parser.parse_args()


def _fetch_url(url, timeout=120):
    import ssl
    from urllib.request import urlopen

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        resp = urlopen(url, context=ctx, timeout=timeout)
    except Exception:
        resp = urlopen(url, timeout=timeout)
    content = resp.read()
    resp.close()
    return content


def download_text_corpus(name, output_path, chunk_size=80):
    print(f"Downloading {name}.zip ...")
    for base in [BASE_URL, FALLBACK_URL]:
        url = base + name + ".zip"
        try:
            content = _fetch_url(url)
            print(f"  Downloaded from {url}")
            break
        except Exception:
            print(f"  Failed: {url}")
    else:
        raise RuntimeError(f"Could not download {name}.zip from any source")

    with zipfile.ZipFile(BytesIO(content), "r") as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"{name}.zip is empty")
        raw = zf.read(names[0]).decode("utf-8", errors="replace")

    words = raw.split()
    print(f"  {len(words):,} words extracted")

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            f.write(" ".join(chunk) + "\n")

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    lines = len(words) // chunk_size + 1
    print(f"  Wrote {lines:,} lines → {output_path} ({size_mb:.1f} MB)")


def download_full_wiki(output_path: str):
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "datasets", "nltk"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    from datasets import load_dataset

    config = "20231101.en"
    print(f"Loading wikimedia/wikipedia ({config}) ... (~20GB download)")
    dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    article_count = 0
    sentence_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item.get("text", "")
            if not text or text.startswith("#REDIRECT"):
                continue
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            for sent in nltk.sent_tokenize(text):
                words = sent.lower().split()
                cleaned = [
                    w.strip(".,;:!?\"'()[]{}<>""''*")
                    for w in words if w.strip(".,;:!?\"'()[]{}<>""''*")
                ]
                if cleaned:
                    f.write(" ".join(cleaned) + "\n")
                    sentence_count += 1
            article_count += 1
            if article_count % 10000 == 0:
                print(f"  {article_count:,} articles, {sentence_count:,} sentences")

    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"\nDone: {article_count:,} articles → {sentence_count:,} sentences ({size_gb:.2f} GB)")


def main():
    args = parse_args()
    output_path = os.path.abspath(args.output)

    mode = args.mode.lower()
    if mode in ("text8", "text9"):
        download_text_corpus(mode, output_path, args.chunk_size)
    elif mode == "full":
        download_full_wiki(output_path)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    print("\nYou can now train with:")
    print(f"  python word2vec.py {output_path} cbow_best.txt cbow checkpoints 3000000 2 1e-5")


if __name__ == "__main__":
    main()

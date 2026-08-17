"""Download full Wikipedia corpus via HuggingFace datasets."""
import os
import re
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

output_path = sys.argv[1] if len(sys.argv) > 1 else 'wiki_full_train.txt'
max_articles = int(sys.argv[2]) if len(sys.argv) > 2 else None

import subprocess
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'datasets', 'nltk', '-q'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

import nltk
for r in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{r}')
    except LookupError:
        nltk.download(r, quiet=True)

from datasets import load_dataset

print('Loading wikimedia/wikipedia (20231101.en) ...')
ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)

PUNCT_STRIP = ".,;:!?\"'()[]{}<>" + chr(0xAB) + chr(0xBB) + chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019) + '*'

count = 0
sc = 0
with open(output_path, 'w', encoding='utf-8') as f:
    for item in ds:
        text = item.get('text', '')
        if not text or text.startswith('#REDIRECT'):
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        for sent in nltk.sent_tokenize(text):
            words = sent.lower().split()
            cleaned = [w.strip(PUNCT_STRIP) for w in words]
            cleaned = [w for w in cleaned if w]
            if cleaned:
                f.write(' '.join(cleaned) + '\n')
                sc += 1
        count += 1
        if count % 10000 == 0:
            size_mb = os.path.getsize(output_path) / (1024 ** 2)
            print(f'{count} articles, {sc} sentences, {size_mb:.0f} MB', flush=True)
        if max_articles and count >= max_articles:
            break

size_gb = os.path.getsize(output_path) / (1024 ** 3)
print(f'Done: {count} articles, {sc} sentences, {size_gb:.2f} GB')

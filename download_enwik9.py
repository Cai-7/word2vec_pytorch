"""Convert enwik9 Wikipedia XML to sentence-per-line text.

Input:  enwik9/enwik9 (1GB Wikipedia XML, must be decompressed first)
Output: enwiki_train.txt (~1.5GB clean text, ~250M words)
"""
import os
import re
import sys
import xml.parsers.expat

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "enwiki_train.txt"
MAX_ARTICLES = int(sys.argv[2]) if len(sys.argv) > 2 else None
XML_PATH = os.path.join("enwik9", "enwik9")

if not os.path.exists(XML_PATH):
    print(f"Error: {XML_PATH} not found. Please decompress enwik9.zip first.")
    print("  The enwik9/ directory should contain the 'enwik9' XML file (1GB).")
    sys.exit(1)

PUNCT = re.compile(r'^[^a-z0-9]+|[^a-z0-9]+$')

article_count = 0
sentence_count = 0
current_title = None
current_text = []
in_page = False
in_title = False
in_text = False
buf = ""


def flush_article():
    global article_count, sentence_count, current_title, current_text
    if not current_title or not current_text:
        current_text = []
        return
    text = "".join(current_text)
    current_text = []
    title = current_title
    current_title = None
    text = text.strip()
    if not text or text.startswith("#REDIRECT") or text.startswith("#redirect"):
        return
    lines = []
    for sent in re.split(r'(?<=[.!?\n])\s+', text):
        sent = sent.strip()
        if not sent:
            continue
        words = [PUNCT.sub('', w.lower()) for w in sent.split()]
        words = [w for w in words if w]
        if words:
            lines.append(" ".join(words))
    if lines:
        output_file.write("\n".join(lines) + "\n")
        sentence_count += len(lines)
    article_count += 1
    if article_count % 10000 == 0:
        size_mb = output_file.tell() / 1024 / 1024
        print(f"  {article_count} articles, {sentence_count} sentences, {size_mb:.0f} MB", flush=True)


def start_element(name, attrs):
    global buf, in_page, in_title, in_text
    buf = ""
    if name == "page":
        in_page = True
    elif name == "title" and in_page:
        in_title = True
    elif name == "text" and in_page:
        in_text = True


def end_element(name):
    global buf, current_title, current_text, in_page, in_title, in_text
    if name == "title" and in_title:
        current_title = buf
        in_title = False
    elif name == "text" and in_text:
        current_text.append(buf)
        in_text = False
    elif name == "page" and in_page:
        flush_article()
        in_page = False


def char_data(data):
    global buf
    if in_title or in_text:
        buf += data


output_file = open(OUTPUT, "w", encoding="utf-8")

print(f"Parsing {XML_PATH} ({os.path.getsize(XML_PATH) / 1024 / 1024:.0f} MB) ...")

p = xml.parsers.expat.ParserCreate()
p.StartElementHandler = start_element
p.EndElementHandler = end_element
p.CharacterDataHandler = char_data

with open(XML_PATH, "rb") as fh:
    while True:
        chunk = fh.read(2 * 1024 * 1024)
        if not chunk:
            break
        try:
            p.Parse(chunk, False)
        except xml.parsers.expat.ExpatError:
            article_count += 1
            break

try:
    p.Parse(b"", True)
except xml.parsers.expat.ExpatError:
    pass

output_file.close()
size_gb = os.path.getsize(OUTPUT) / (1024 ** 3)
print(f"\nDone: {article_count} articles → {sentence_count} sentences → {size_gb:.2f} GB")
print(f"Output: {OUTPUT}")

"""Export word embeddings from a saved checkpoint."""
import sys
import torch
from model import CBOWModel, SkipGramModel
from input_data import InputData

CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else r"checkpoints_enwik9\checkpoint_00440477.pt"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "word_embedding.txt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model_type = checkpoint.get("model_type", "cbow")
emb_dimension = checkpoint.get("emb_dimension", 300)
input_file = checkpoint.get("input_file_name", "enwiki_train.txt")
min_count = checkpoint.get("min_count", 5)

print(f"[1/4] Reading input file: {input_file}")
data = InputData(input_file, min_count)
emb_size = len(data.word2id)

print(f"[2/4] Building model ({model_type}, dim={emb_dimension}, vocab={emb_size})")
if model_type == "cbow":
    model = CBOWModel(emb_size, emb_dimension)
else:
    model = SkipGramModel(emb_size, emb_dimension)

print(f"[3/4] Loading checkpoint weights (step={checkpoint.get('step')})")
model.load_state_dict(checkpoint["model_state_dict"])

print(f"[4/4] Saving embeddings to: {OUTPUT_PATH}")
model.save_embedding(data.id2word, OUTPUT_PATH, use_cuda=False)
print("Done!")

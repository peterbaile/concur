import json
import yaml

def read_json(fn):
    with open(fn) as f:
       return json.load(f)

def write_json(obj, fn):
    with open(fn, 'w') as f:
        json.dump(obj, f, indent=2)

def read_yaml(fn):
    with open(fn) as f:
        return yaml.safe_load(f)

def embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
    ).cuda()

    return model.encode(texts, show_progress_bar=True, batch_size=16, convert_to_tensor=True)
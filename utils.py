import json
import yaml
import pandas as pd

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

def get_all_strategies():
    models = ["llama3b", "llama8b", "qwen2b", "qwen3b", "qwen7b"]

    strategies = []
    for model in models:
        for decoding in ["vanilla", "cot"]:
            strategies.append((model, decoding, f"{model}_{decoding}"))

    return strategies

def eval_strategies(test_data_path, optimal_strategies, metadata):
    """
    Evaluate the optimal_strategies to output the final outcomes

    optimal_strategies: list[(qid, model, strategy)]
    """
    
    df = pd.read_csv(test_data_path)
    info_dict = {}
    for _, row in df.iterrows():
        info_dict[(row["question_id"], row["model"], row["decoding"])] = [
            row["label"],
            row["flops"],
        ]

    acc_list, cost_list = [], []

    for decision in optimal_strategies:
        acc, cost = info_dict[decision]
        acc_list.append(acc)
        cost_list.append(cost)

    avg_acc = sum(acc_list) / len(acc_list)
    avg_cost = sum(cost_list) / len(cost_list)

    print(f"{metadata} | number of tasks: {len(acc_list):<5} | accuracy: {avg_acc:<6.3f} | cost: {avg_cost:<6.2f}")
    return avg_acc, avg_cost
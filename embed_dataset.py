import argparse
import torch
from transformers import set_seed

from utils import embed, read_yaml, read_json

def embed_task(task_path, save_path):
    tasks = read_json(task_path)

    task_texts = []

    for task in tasks:
        choice_map = {
            0: "(A)",
            1: "(B)",
            2: "(C)",
            3: "(D)",
            4: "(E)",
            5: "(F)",
            6: "(G)",
            7: "(H)",
            8: "(I)",
            9: "(J)",
            10: "(K)",
        }

        choices = task['options']
        choices = "\n".join(
                [f"{choice_map[i]} {choice}" for i, choice in enumerate(choices)]
            )
        task_text = f"Question: {task['question']}\n\nChoices:\n{choices}"
        task_texts.append(task_text)
    
    task_embeddings = embed(task_texts)

    torch.save(task_embeddings, save_path)

    print(f'Embedded {len(tasks)} tasks and saved to {save_path}')

def embed_strategy(strategy_path, save_path):
    # embed strategies (models and decoding methods)
    strategies_json = read_json(strategy_path)
    strategies_embed = {'model': {}, 'decoding': {}}

    for model in strategies_json['model']:
        strategies_embed["model"][model] = embed([strategies_json["model"][model]])[0]

    for decoding in strategies_json["decoding"]:
        strategies_embed["decoding"][decoding] = embed([strategies_json["decoding"][decoding]])[0]

    torch.save(strategies_embed, save_path)

def main():
    set_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yml')
    args = parser.parse_args()

    config = read_yaml(args.config_path)

    embed_task(config['task_path'], config['task_emb_path'])
    embed_strategy(config['strategy_path'], config['strategy_emb_path'])

if __name__ == '__main__':
    main()
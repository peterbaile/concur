import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from utils import get_all_strategies, eval_strategies, read_yaml

def get_optimal_strategies(output_path, weight):
    '''Decide the optimal policy based on the routing table'''

    df_list = []
    for strategy in get_all_strategies():
        df_list.append(
            pd.read_csv(Path(output_path) / f'pred_{strategy[-1]}.csv')
        )
    df = pd.concat(df_list)

    # construct the routing table
    routing_table = {}
    for _, row in df.iterrows():
        task_id, model, decoding = row["question_id"], row["model"], row["decoding"]
        reward, cost = row["pred_cls_prob"], row["pred_flops"]

        if task_id not in routing_table:
            routing_table[task_id] = {}

        routing_table[task_id][(model, decoding)] = weight * reward - (1 - weight) * cost

    optimal_strategies = []
    for task_id in routing_table:
        scores = []
        strategies = list(routing_table[task_id].keys())
        for strategy in strategies:
            scores.append(routing_table[task_id][strategy])
        
        scores_np = np.array(scores)
        optimal_strategy_idx = np.argmax(scores_np)
        optimal_model, optimal_decoding = strategies[optimal_strategy_idx]
        
        optimal_strategies.append(
            (task_id, optimal_model, optimal_decoding)
        ) 

    return optimal_strategies


def pareto(test_data_path, output_path, weights: list[float]):
    for weight in weights:
        optimal_strategies = get_optimal_strategies(
            output_path, weight
        )
        eval_strategies(test_data_path, optimal_strategies, metadata=f'weight: {weight:<8}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yml')
    parser.add_argument('--weight', nargs='+', default=None, help='preference between accuracy and cost')
    args = parser.parse_args()

    config = read_yaml(args.config_path)
    if args.weight:
        weights = [float(weight) for weight in args.weight]
    else:
        weights = [
            0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 1
        ]
    pareto(config['test_data_path'], config['output_path'], weights)

if __name__ == "__main__":
    main()

import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Literal
import argparse
from pathlib import Path

from utils import read_yaml, eval_strategies, get_all_strategies

def dp_routing(tasks, budget, scale=1):
    """
    Assign a batch of tasks a total `budget`.

    dp[i][b]: The value is the maximum reward achievable using options from groups 1 to i and budget ≤ b
    time complexity: O(m * B * n) — efficient if budget B is not too large (up to ~10⁵)

    dp[i][b] = max_{j in options} (db[i-1][b - cost_j] + reward_j)

    Args:
        tasks: list of tasks. Each list of tasks is a list of (cost, reward) tuples of all strategies (floats allowed)
        budget: float total cost budget
        scale: factor to convert float costs to integers (default 2 decimal places)

    Returns:
        max_reward: float
        chosen_indices: list of selected option indices, one per group
    """
    # Scale costs and budget to integers
    int_tasks = [
        [(int(cost * scale + 0.5), reward) for cost, reward in task]
        for task in tasks
    ]
    int_budget = int(budget * scale + 0.5)

    m = len(tasks)
    dp = [float("-inf")] * (int_budget + 1)
    dp[0] = 0

    # For traceback
    path = [[None] * (int_budget + 1) for _ in range(m)]

    cumulative_min_costs, cumulative_max_costs = [], []
    min_total, max_total = 0, 0
    for group in int_tasks:
        group_min = min(cost for cost, _ in group)
        group_max = max(cost for cost, _ in group)
        min_total += group_min
        max_total += group_max
        cumulative_min_costs.append(min_total)
        cumulative_max_costs.append(max_total)

    for i, group in enumerate(tqdm(int_tasks)):
        new_dp = [float("-inf")] * (int_budget + 1)

        effective_max_b = min(int_budget, cumulative_max_costs[i])

        for opt_idx, (cost, reward) in enumerate(group):
            effective_min_b = max(cost, cumulative_min_costs[i])

            # for b in range(int_budget, cost - 1, -1):
            for b in range(effective_max_b, effective_min_b - 1, -1):
                prev_b = b - cost
                if dp[prev_b] != float("-inf"):
                    new_reward = dp[prev_b] + reward
                    if new_reward > new_dp[b]:
                        new_dp[b] = new_reward
                        path[i][b] = (prev_b, opt_idx)

        dp = new_dp

    # Find best final reward and budget used
    best_reward = max(dp)
    best_b = dp.index(best_reward)

    # Traceback to find selected indices
    chosen_indices = [None] * m
    b = best_b
    for i in reversed(range(m)):
        prev_b, opt_idx = path[i][b]
        chosen_indices[i] = opt_idx
        b = prev_b

    return best_reward, chosen_indices


def equal_routing(tasks, budget_per_task):
    """
    Assign each question exactly `budget_per_task`.

    Args:
        tasks: list of tasks. Each list of tasks is a list of (cost, reward) tuples of all strategies (floats allowed)
        budget_per_task: cost budget for each task
    """
    optimal_strategy_idxs = []

    for task in tasks:
        # zero out all options falling within the budget
        filtered_strategies = [
            strategy if strategy[0] <= budget_per_task else [-float('inf'), -float('inf')] for strategy in task
        ]
        assert len(filtered_strategies) == len(task)

        if all(strategy == [-100, -100] for strategy in filtered_strategies):
            # no strategy within budget --> choose the one with the smallest budget
            optimal_strategy_idx = np.argmin([x[0] for x in task])
        else:
            # choose the one with the highest accuracy
            optimal_strategy_idx = np.argmax([x[1] for x in filtered_strategies])

        optimal_strategy_idxs.append(optimal_strategy_idx)

    return None, optimal_strategy_idxs

def split_batches(lists, num_batches, batch_idx):
    interval = len(lists) // num_batches + 1
    start, end = batch_idx * interval, (batch_idx + 1) * interval
    return lists[start : end]

def get_optimal_strategies(
    mode: Literal["local", "global"], budget_per_q, num_batches, batch_idx,
    output_path
):
    strategies = [strategy[-1] for strategy in get_all_strategies()]
    
    df = pd.read_csv(Path(output_path) / f'pred_{strategies[0]}.csv')
    num_tasks = df.shape[0]
    task_ids = df["question_id"].tolist()
    
    # len(tasks) x len(strategies) x 2
    # for each task, each list includes outcomes (cost, success probability) of all strategies
    tasks = [[] for _ in range(num_tasks)]

    for strategy in strategies:
        df = pd.read_csv(Path(output_path) / f'pred_{strategy}.csv')
        for task_idx, row in df.iterrows():
            tasks[task_idx].append([row['pred_flops'], row['pred_cls_prob']])

    # group tasks by batches
    assert len(task_ids) == len(tasks)
    task_ids = split_batches(task_ids, num_batches, batch_idx)
    tasks = split_batches(tasks, num_batches, batch_idx)
    assert len(task_ids) == len(tasks)

    num_tasks = len(tasks)
    print(f"Routing {num_tasks} tasks using {mode} optimization...")

    if mode == 'global':
        _, optimal_strategy_idxs = dp_routing(tasks, budget_per_q * num_tasks)
    elif mode == 'local':
        _, optimal_strategy_idxs = equal_routing(tasks, budget_per_q)

    optimal_strategies = []
    for task_id, optimal_strategy_idx in zip(task_ids, optimal_strategy_idxs):
        model, decoding = strategies[optimal_strategy_idx].split('_')
        optimal_strategies.append((task_id, model, decoding))

    return optimal_strategies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yml')
    parser.add_argument("--num_batches", type=int, default=1, help='number of batches to optimize over')
    parser.add_argument("--budget", type=int, help='cost budget per each task', required=True)
    parser.add_argument("--mode", choices=['local', 'global'], required=True)
    args = parser.parse_args()

    config = read_yaml(args.config_path)

    optimal_strategies = []
    for batch_idx in range(args.num_batches):
        optimal_strategies += get_optimal_strategies(
            args.mode, args.budget, args.num_batches, batch_idx, config['output_path']
        )

    eval_strategies(
        config['test_data_path'], optimal_strategies, f'budget per task: {args.budget:<5}'
    )

if __name__ == "__main__":
    main()
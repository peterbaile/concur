#  CONCUR: A Framework for Continual Constrained and Unconstrained Routing

[![Paper](https://img.shields.io/badge/OpenReview-ICLR%202026-blue)](https://openreview.net/pdf?id=gCUY6QIv8r)

If you find our code, or the paper helpful, please cite the paper
```
@article{chen2025concur,
  title={CONCUR: A Framework for Continual Constrained and Unconstrained Routing},
  author={Chen, Peter Baile and Li, Weiyue and Roth, Dan and Cafarella, Michael and Madden, Samuel and Andreas, Jacob},
  journal={arXiv preprint arXiv:2512.09386},
  year={2025}
}
```

## Code walkthrough

This repo walks you through an example using the [MMLU dataset](https://arxiv.org/abs/2009.03300).
Download the data from [this Google Drive link](https://drive.google.com/drive/folders/1RUdhVdNJjGYQxNG9t5An92d2DLz5Sfy5) and put it in `data`.
```
data/
    mmlu/
        task.json
        test.csv
        ...
    strategies.json             # Descriptions of computational strategies (models and decoding methods)
config.yml                      # Path specification
embed_dataset.py                # Embed tasks and strategies
predictor.py                    # Training and inference of predictors
routing_constrained.py          # Constrained routing
routing_unconstrained.py        # Unconstrained routing
```

**1. Pre-processing**

This step embeds the input tasks and computation strategies using an off-the-shelf text embedding model to generate the general-purpose reprensetations.
```
python embed_dataset.py
```

**2. Predictor training**

This step trains the accuracy and cost estimators for strategies using the train and validation data.

```
# train predictors for qwen7b_cot and llama3b_vanilla
python predictor.py --strategy qwen7b_cot llama3b_vanilla

# train predictors for all strategies
python predictor.py --strategy all
```

**3. Predictor inference**

This step runs the trained predictors on the test data.

```
# run predictors for qwen7b_cot and llama3b_vanilla
python predictor.py --strategy qwen7b_cot llama3b_vanilla --predict

# run predictors for all strategies
python predictor.py --strategy all --predict
```

**Unconstrained routing**

This step performs unconstrained routing based on the predictor outputs.
* `weight` refers to the preference between accuracy and cost (higher `weight` means higher emphasis on accuracy).

```
# Run unconstrained routing on specified weights
python routing_unconstrained.py --weight 0.99 0.999

# Run unconstrained routing on the default weights
python routing_unconstrained.py
```

**Constrained routing**

This step performs constrained routing based on the predictor outputs.
* `budget` refers to the maximum cost budget allowed for each task.
* `mode=global` (our approach): distributes the total budget jointly across all tasks
* `mode=local` (baseline approach): treats each task independently and allocates the budget evenly across tasks

```
# Run constrained routing using global optimization
python routing_constrained.py --budget 40 --mode global

# Run constrained routing using local optimization
python routing_constrained.py --budget 40 --mode local
```

## Contact
If you have any questions or feedback, please send an email to peterbc@mit.edu.
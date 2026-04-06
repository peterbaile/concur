#  CONCUR: A Framework for Continual Constrained and Unconstrained Routing

If you find our code, or the paper helpful, please cite the paper
```
@article{chen2025concur,
  title={CONCUR: A Framework for Continual Constrained and Unconstrained Routing},
  author={Chen, Peter Baile and Li, Weiyue and Roth, Dan and Cafarella, Michael and Madden, Samuel and Andreas, Jacob},
  journal={arXiv preprint arXiv:2512.09386},
  year={2025}
}
```

## Execution

This repo walks you through an example using the [MMLU dataset](https://arxiv.org/abs/2009.03300).
Download the data from [this Google Drive link](https://drive.google.com/drive/folders/1RUdhVdNJjGYQxNG9t5An92d2DLz5Sfy5) and put it in `data`.
```
data/
    mmlu/
        task.json
        test.csv
        ...
```

**Pre-processing**

This step embeds the input tasks and computation strategies using an off-the-shelf text embedding model to generate the general-purpose reprensetations.
```
python embed_dataset.py --config_path config.yml
```

**Predictor training**

This step trains the accuracy and cost estimators for a strategy (e.g., `qwen7b_cot`) using the train and validation data.

```
python predictor.py --config_path config.yml --strategy qwen7b_cot
```

**Predictor inference**

This step runs the trained predictors on the test data.

```
python predictor.py --config_path config.yml --strategy qwen7b_cot --predict
```

**Unconstrained routing**



**Constrained routing**



## Contact
If you have any questions or feedback, please send an email to peterbc@mit.edu.
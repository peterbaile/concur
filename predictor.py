import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import set_seed
import time
import yaml
from pathlib import Path
import os

def prepare_data(data_path, strategy_emb_path, task_emb_path, strategy):
    # Create ID mappings for models and decoding methods (e.g., llama8b_vanilla)
    model, decoding = strategy.split("_")
    models, decodings = [model], [decoding]
    model2id = {name: idx for idx, name in enumerate(models)}
    decoding2id = {stype: idx for idx, stype in enumerate(decodings)}

    df = pd.read_csv(data_path)
    df = df[(df["model"] == model) & (df["decoding"] == decoding)]
    assert len(df) > 0, f'No data for strategy {strategy}'
    task_embeddings = torch.load(task_emb_path)
    strategy_embeddings = torch.load(strategy_emb_path)

    X, y_cls, y_reg, model_ids, decoding_ids = [], [], [], [], []

    for _, row in df.iterrows():
        task_emb = task_embeddings[row["question_id"]]
        model_emb = strategy_embeddings["model"][model]
        decoding_emb = strategy_embeddings["decoding"][decoding]
        combined_emb = torch.hstack([model_emb, decoding_emb, task_emb])
        
        X.append(combined_emb)
        y_cls.append(row["label"])
        y_reg.append(row["flops"])

        model_ids.append(model2id[model])
        decoding_ids.append(decoding2id[decoding])

    return (
        torch.vstack(X),
        np.array(y_cls),
        np.array(y_reg),
        np.array(model_ids),
        np.array(decoding_ids),
        model2id,
        decoding2id,
        len(models),
        len(decodings),
    )


class DualTargetDataset(Dataset):
    def __init__(self, X, y_cls, y_reg, model_ids, decoding_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.model_ids = torch.tensor(model_ids, dtype=torch.long)
        self.decoding_ids = torch.tensor(decoding_ids, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_cls[idx],
            self.y_reg[idx],
            self.model_ids[idx],
            self.decoding_ids[idx],
        )


class MLPClassifier(nn.Module):
    def __init__(
        self, input_dim, num_models, num_decodings, emb_dim=64, hidden_dim=100,
        num_classes=2,
    ):
        super().__init__()
        self.model_emb = nn.Embedding(num_models, emb_dim)
        self.decoding_emb = nn.Embedding(num_decodings, emb_dim)
        self.emb_dim = emb_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim + 3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.q_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, model_ids, decoding_ids):
        model_emb = self.model_emb(model_ids)  # Shape: (batch_size, emb_dim)
        decoding_emb = self.decoding_emb(decoding_ids)  # Shape: (batch_size, emb_dim)

        assert x.shape[1] == 3 * self.emb_dim

        # L36: combined_emb = torch.hstack([model_emb, decoding_emb, task_emb])
        # x[:, self.emb_dim * 2 : self.emb_dim * 3] refers to the general-purpose representation of task_emb
        _q_emb = self.q_proj(x[:, self.emb_dim * 2 : self.emb_dim * 3])
        x = torch.cat([x, model_emb, decoding_emb, _q_emb], dim=-1)

        return self.model(x)


class MLPRegressor(nn.Module):
    def __init__(
        self, input_dim, num_models, num_decodings, emb_dim=64, hidden_dim=100
    ):
        super().__init__()
        self.model_emb = nn.Embedding(num_models, emb_dim)
        self.decoding_emb = nn.Embedding(num_decodings, emb_dim)
        self.emb_dim = emb_dim

        self.model = nn.Sequential(
            nn.Linear(
                input_dim + 3 * emb_dim, hidden_dim
            ),  # Adjust input_dim for trainable embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.q_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, model_ids, decoding_ids):
        model_emb = self.model_emb(model_ids)  # Shape: (batch_size, emb_dim)
        decoding_emb = self.decoding_emb(decoding_ids)  # Shape: (batch_size, emb_dim)

        assert x.shape[1] == 3 * self.emb_dim

        _q_emb = self.q_proj(x[:, self.emb_dim * 2 : self.emb_dim * 3])
        x = torch.cat([x, model_emb, decoding_emb, _q_emb], dim=-1)

        return self.model(x).squeeze(1)


def evaluate_classifier(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb_cls, _, model_ids, decoding_ids in val_loader:
            xb, yb_cls = xb.to(device), yb_cls.to(device)
            model_ids, decoding_ids = model_ids.to(device), decoding_ids.to(device)
            preds = model(xb, model_ids, decoding_ids)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb_cls).sum().item()
            total += yb_cls.size(0)

    overall_acc = correct / total if total > 0 else 0.0
    print(f"\nClassifier Accuracy: {overall_acc:.2%}")
    return overall_acc


def evaluate_regressor(model, val_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, _, yb_reg, model_ids, decoding_ids in val_loader:
            xb = xb.to(device)
            model_ids, decoding_ids = model_ids.to(device), decoding_ids.to(device)
            pred = model(xb, model_ids, decoding_ids).cpu()
            preds.append(pred)
            targets.append(yb_reg)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = nn.MSELoss()(torch.tensor(preds), torch.tensor(targets))

    print(f"Regressor MSE: {mse.item():.4f}")

    return mse.item()


def train_dual_models(
    X_train, y_cls_train, y_reg_train,
    model_ids_train, decoding_ids_train,
    X_val, y_cls_val, y_reg_val,
    model_ids_val, decoding_ids_val,
    num_models,
    num_decodings,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    patience=5,
    save_cls="best_model_cls1.pt",
    save_reg="best_model_reg1.pt",
    cls_hidden_dim=768,
    reg_hidden_dim=768,
    emb_dim=768,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DualTargetDataset(
        X_train, y_cls_train, y_reg_train, model_ids_train, decoding_ids_train
    )
    val_dataset = DualTargetDataset(
        X_val, y_cls_val, y_reg_val, model_ids_val, decoding_ids_val
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_cls_train.tolist()))

    model_cls = MLPClassifier(
        input_dim,
        num_models,
        num_decodings,
        emb_dim=emb_dim,
        hidden_dim=cls_hidden_dim,
        num_classes=num_classes,
    ).to(device)
    model_reg = MLPRegressor(
        input_dim,
        num_models,
        num_decodings,
        emb_dim=emb_dim,
        hidden_dim=reg_hidden_dim,
    ).to(device)

    optimizer_cls = torch.optim.AdamW(model_cls.parameters(), lr=lr)
    optimizer_reg = torch.optim.AdamW(model_reg.parameters(), lr=lr)

    loss_cls_fn = nn.CrossEntropyLoss()
    loss_reg_fn = nn.MSELoss()

    best_acc, best_mse = 0, float("inf")
    patience_counter_cls = 0
    patience_counter_reg = 0
    stop_cls, stop_reg = False, False

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        total_loss_cls, total_loss_reg = 0, 0
        model_cls.train()
        model_reg.train()

        for xb, yb_cls, yb_reg, model_ids, decoding_ids in train_loader:
            xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
            model_ids, decoding_ids = model_ids.to(device), decoding_ids.to(device)

            if not stop_cls:
                optimizer_cls.zero_grad()
                pred_cls = model_cls(xb, model_ids, decoding_ids)
                loss_cls = loss_cls_fn(pred_cls, yb_cls)
                loss_cls.backward()
                optimizer_cls.step()
                total_loss_cls += loss_cls.item()

            if not stop_reg:
                optimizer_reg.zero_grad()
                pred_reg = model_reg(xb, model_ids, decoding_ids)
                loss_reg = loss_reg_fn(pred_reg, yb_reg)
                loss_reg.backward()
                optimizer_reg.step()
                total_loss_reg += loss_reg.item()

        avg_loss_cls = total_loss_cls / len(train_loader)
        avg_loss_reg = total_loss_reg / len(train_loader)
        print(
            f"  - Avg Loss_cls: {avg_loss_cls:.4f} | Avg Loss_reg: {avg_loss_reg:.4f}"
        )

        if epoch % 3 == 0:
            if not stop_cls:
                acc = evaluate_classifier(model_cls, val_loader, device)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model_cls.state_dict(), save_cls)
                    print(f"Saved best classifier model with accuracy {best_acc:.2%}")
                    patience_counter_cls = 0
                else:
                    patience_counter_cls += 1
                    print(f"Classifier patience {patience_counter_cls}/{patience}")
                    if patience_counter_cls >= patience:
                        print("Classifier early stopping triggered.")
                        stop_cls = True

            if not stop_reg:
                mse = evaluate_regressor(model_reg, val_loader, device)
                if mse < best_mse:
                    best_mse = mse
                    torch.save(model_reg.state_dict(), save_reg)
                    print(f"Saved best regressor model with MSE {best_mse:.4f}")
                    patience_counter_reg = 0
                else:
                    patience_counter_reg += 1
                    print(f"Regressor patience {patience_counter_reg}/{patience}")
                    if patience_counter_reg >= patience:
                        print("Regressor early stopping triggered.")
                        stop_reg = True

        if stop_cls and stop_reg:
            print("Both models early stopped. Ending training loop.")
            break


def predict(
    data_path,
    task_emb_path,
    strategy_embedding_path,
    cls_model_path,
    reg_model_path,
    output_path,
    strategy,
    device=None,
    cls_hidden_dim=768,
    reg_hidden_dim=768,
    emb_dim=768,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, decoding = strategy.split("_")

    df = pd.read_csv(data_path)
    df = df[(df["model"] == model) & (df["decoding"] == decoding)]
    assert len(df) > 0, f'No data for strategy {strategy}'

    task_embeddings = torch.load(task_emb_path)
    strategy_embeddings = torch.load(strategy_embedding_path)
    model_embeddings = strategy_embeddings["model"]
    decoding_embeddings = strategy_embeddings["decoding"]

    # Create ID mappings
    models, decodings = [model], [decoding]
    model2id = {name: idx for idx, name in enumerate(models)}
    decoding2id = {stype: idx for idx, stype in enumerate(decodings)}
    num_models, num_decodings = len(models), len(decodings)

    example_input = torch.hstack(
        [
            next(iter(model_embeddings.values())),
            next(iter(decoding_embeddings.values())),
            task_embeddings[0],
        ]
    )
    input_dim = example_input.shape[0]

    cls_model = MLPClassifier(
        input_dim=input_dim,
        num_models=num_models,
        num_decodings=num_decodings,
        emb_dim=emb_dim,
        hidden_dim=cls_hidden_dim,
        num_classes=2,
    ).to(device)
    reg_model = MLPRegressor(
        input_dim=input_dim,
        num_models=num_models,
        num_decodings=num_decodings,
        emb_dim=emb_dim,
        hidden_dim=reg_hidden_dim,
    ).to(device)
    cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
    cls_model.eval()
    reg_model.eval()

    def process_single_prediction(args):
        (
            question_id,
            task_emb,
            model_name,
            model_emb,
            decoding_name,
            decoding_emb,
            gt_label,
            gt_flops,
        ) = args
        input_vector = torch.hstack([model_emb, decoding_emb, task_emb])
        input_tensor = (
            torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
        )
        model_id = torch.tensor([model2id[model_name]], dtype=torch.long).to(device)
        decoding_id = torch.tensor([decoding2id[decoding_name]], dtype=torch.long).to(device)

        with torch.no_grad():
            out_cls = cls_model(input_tensor, model_id, decoding_id)
            probs = torch.softmax(out_cls, dim=-1)
            pred_label = torch.argmax(out_cls, dim=1).item()
            out_reg = reg_model(input_tensor, model_id, decoding_id)
            pred_length = out_reg.item()

        pred_prob_1 = probs[0, 1].item()

        return (question_id, model_name, decoding_name), {
            "gt_label": gt_label,
            "pred_label": pred_label,
            "pred_cls_prob": pred_prob_1,
            "gt_flops": gt_flops,
            "pred_flops": pred_length,
        }

    tasks = []
    for _, row in df.iterrows():
        task_emb = task_embeddings[row["question_id"]]
        assert row['model'] == model and row['decoding'] == decoding
        model_emb, decoding_emb = model_embeddings[row["model"]], decoding_embeddings[row["decoding"]]
        tasks.append((
            row["question_id"], task_emb,
            model, model_emb,
            decoding, decoding_emb,
            row["label"], row["flops"],
        ))

    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_task = {
            executor.submit(process_single_prediction, task): task for task in tasks
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Processing predictions",
        ):
            key, result = future.result()
            results[key] = result

    results_list = []
    for (question_id, model_name, decoding_name), res in sorted(results.items()):
        results_list.append(
            {
                "question_id": question_id,
                "model": model_name,
                "decoding": decoding_name,
                **res,
            }
        )

    df_preds = pd.DataFrame(results_list)

    df_preds.to_csv(output_path, index=False)
    print(f"\nSaved prediction results to '{output_path}'")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, help='model_decoding (e.g., llama8b_cot)')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    set_seed(1234)

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    os.makedirs(Path(args.output_path), exist_ok=True)
    save_cls_path = Path(args.output_path) / f'accuracy_cls_{args.strategy}.pt'
    save_reg_path = Path(args.output_path) / f'cost_reg_{args.strategy}.pt'
    save_pred_path = Path(args.output_path) / f'pred_{args.strategy}.csv'

    if args.predict:
        print("Predicting...")
        predict(
            config['test_data_path'], config['task_emb_path'], config['strategy_emb_path'],
            save_cls_path, save_reg_path, save_pred_path, args.strategy
        )
    else:
        start_time = time.time()
        print(f"Loading training and validation data for {args.strategy}...")
        (
            X_train, y_cls_train, y_reg_train,
            model_ids_train, decoding_ids_train,
            _, _,
            num_models_train, num_decodings_train,
        ) = prepare_data(
            config['train_data_path'], config['strategy_emb_path'], config['task_emb_path'], args.strategy
        )
        (
            X_val, y_cls_val, y_reg_val,
            model_ids_val, decoding_ids_val,
            _, _, _, _,
        ) = prepare_data(config['val_data_path'], config['strategy_emb_path'], config['task_emb_path'], args.strategy)

        print(
            f"Training data: X: {X_train.shape}, y_cls: {y_cls_train.shape}, y_reg: {y_reg_train.shape}, "
            f"model_ids: {model_ids_train.shape}, decoding_ids: {decoding_ids_train.shape}"
        )
        print(
            f"Validation data: X: {X_val.shape}, y_cls: {y_cls_val.shape}, y_reg: {y_reg_val.shape}, "
            f"model_ids: {model_ids_val.shape}, decoding_ids: {decoding_ids_val.shape}"
        )

        print(f'Training accuracy and cost predictors for {args.strategy}')
        train_dual_models(
            X_train, y_cls_train, y_reg_train,
            model_ids_train, decoding_ids_train,
            X_val, y_cls_val, y_reg_val,
            model_ids_val, decoding_ids_val,
            num_models_train, num_decodings_train,
            save_cls=save_cls_path, save_reg=save_reg_path
        )

        end_time = time.time()
        train_time = end_time - start_time
        print(f'Training for {args.strategy} completed in {train_time:.3f} seconds')

if __name__ == "__main__":
    main()
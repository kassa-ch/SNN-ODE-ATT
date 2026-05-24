import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPS_ROOT = PROJECT_ROOT / "Scrips"
PUBLIC_DATASET_ROOT = SCRIPS_ROOT / "data_loader" / "public_datasets"
for path in (PROJECT_ROOT, SCRIPS_ROOT, PUBLIC_DATASET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_loader import TimeSeriesDataset, custom_collate
from enhanced_loss import EnhancedSNNLoss
from model_factory import create_model
from public_data_resolver import find_public_label_paths, resolve_public_split


def _loader_shape(paths, apply_poisson_sampling=False):
    from torch.utils.data import DataLoader

    selected = [str(p) for p in paths[:2]]
    dataset = TimeSeriesDataset(selected, apply_poisson_sampling=apply_poisson_sampling)
    loader = DataLoader(dataset, batch_size=len(selected), shuffle=False, collate_fn=custom_collate)
    x, mask, delta_t, file_paths = next(iter(loader))
    return loader, {
        "x": list(x.shape),
        "mask": list(mask.shape),
        "delta_t": list(delta_t.shape),
        "files": [Path(p).name for p in file_paths],
        "inferred_input_dim": int(x.shape[-1]),
    }


def run_check_only(args):
    data_dir = PROJECT_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    normal, abnormal = find_public_label_paths(data_dir)
    apply_poisson = args.view == "nonuniform"
    _loader, shape = _loader_shape(normal, apply_poisson_sampling=apply_poisson)
    split = resolve_public_split(args.dataset, args.view, data_dir, seed=42, save=True)

    model = create_model("PredictiveSNN_ODEATT_Model", input_dim=shape["inferred_input_dim"])
    loss = EnhancedSNNLoss()
    summary = {
        "mode": "check-only",
        "dataset": args.dataset,
        "view": args.view,
        "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
        "normal_count": len(normal),
        "abnormal_count": len(abnormal),
        "dataloader_shape": shape,
        "model_class": model.__class__.__name__,
        "loss_class": loss.__class__.__name__,
        "split_manifest": split.get("manifest_path"),
        "trained": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_smoke(args):
    from torch.utils.data import DataLoader
    from train import model_forward

    data_dir = PROJECT_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    normal, abnormal = find_public_label_paths(data_dir)
    if not normal:
        raise FileNotFoundError(f"No normal CSV files in {data_dir}")

    apply_poisson = args.view == "nonuniform"
    split = resolve_public_split(args.dataset, args.view, data_dir, seed=42, save=True)
    train_paths = [PROJECT_ROOT / p for p in split["train_normal_paths"][: max(args.batch_size * args.max_batches, 1)]]
    dataset = TimeSeriesDataset([str(p) for p in train_paths], apply_poisson_sampling=apply_poisson)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    x0, _m0, _dt0, _paths0 = next(iter(loader))

    device = torch.device(args.device)
    model = create_model("PredictiveSNN_ODEATT_Model", input_dim=int(x0.shape[-1])).to(device)
    criterion = EnhancedSNNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    batches_seen = 0
    last_loss = None
    for _epoch in range(args.epochs):
        for x, mask, delta_t, _paths in loader:
            if batches_seen >= args.max_batches:
                break
            x = x.to(device)
            mask = mask.to(device)
            delta_t = delta_t.to(device)
            recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq = model_forward(
                model,
                "PredictiveSNN_ODEATT_Model",
                x,
                mask=mask,
                delta_t=delta_t,
            )
            loss, _info = criterion(
                reconstructions=recon,
                predictions=pred,
                x=x,
                mask=mask,
                z_mean=z_mean,
                mem_seq=mem_seq,
                mem_reset_seq=mem_reset_seq,
                spike_seq=spike_seq,
                rate_seq=rate_seq,
                h_seq=h_seq,
                delta_ts=delta_t,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            batches_seen += 1
        if batches_seen >= args.max_batches:
            break

    out_dir = PROJECT_ROOT / "Results" / "models" / "checkpoints" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{args.dataset.lower().replace('-', '_')}_{args.view}_snnodeatt_smoke.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "PredictiveSNN_ODEATT_Model",
        "input_dim": int(x0.shape[-1]),
        "epochs": args.epochs,
        "batches_seen": batches_seen,
        "last_loss": last_loss,
        "dataset": args.dataset,
        "view": args.view,
    }, checkpoint_path)
    print(json.dumps({
        "mode": "smoke",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "epochs": args.epochs,
        "batches_seen": batches_seen,
        "last_loss": last_loss,
        "normal_count": len(normal),
        "abnormal_count": len(abnormal),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HAI", "ST-AWFD", "TEP"], required=True)
    parser.add_argument("--view", choices=["uniform", "nonuniform"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check_only:
        run_check_only(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()

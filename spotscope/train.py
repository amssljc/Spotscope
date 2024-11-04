import os
import torch
from torch import nn
from tqdm import tqdm

from .models import CLIPModel, BLEEP, ST_NET
from .utils import AvgMeter, get_lr, setup_seed
from .dataset import build_train_loaders


def train_epoch(model, train_loader, optimizer, device="cuda:0"):
    loss_meter = AvgMeter()
    for batch in train_loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["image", "annotations", "relative_coords"]
        }
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

    return loss_meter.avg  # Return average loss for the epoch


def test_epoch(model, test_loader, device="cuda:0"):
    loss_meter = AvgMeter()
    for batch in test_loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["image", "annotations", "relative_coords"]
        }
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

    return loss_meter.avg  # Return average loss for the epoch


def train(
    exp_name,
    dataset_paths,
    batch_size,
    max_epochs=150,
    seed=42,
    lr=1e-3,
    model=CLIPModel(),
    task="discrete",
    train_ratio=0.9,
    device="cuda:0",
    weight_decay=1e-4,
    patience=20,
):
    print(f"Setting Seed: {seed}")
    setup_seed(seed)
    # torch.cuda.set_device(7)
    print("Starting training...")

    model = model.to(device)
    # Prepare data loaders
    train_loader, test_loader = build_train_loaders(
        batch_size, dataset_paths, train_ratio, task=task
    )

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_epoch = 0
    test_loss = float("inf")
    tqdm_object = tqdm(range(max_epochs), total=max_epochs)
    for epoch in tqdm_object:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device)

        model.eval()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_loss = test_epoch(model, test_loader, device)

        tqdm_object.set_description(f"Epoch {epoch+1}/{max_epochs}")
        tqdm_object.set_postfix(train_loss=train_loss, valid_loss=test_loss)

        # Save the best model
        if test_loss < best_loss:
            best_loss = test_loss
            exp_fold_name = "/".join(exp_name.split("/")[:-1])
            os.makedirs(exp_fold_name, exist_ok=True)
            print("Saving Best Model! Loss:", best_loss)
            torch.save(model.state_dict(), f"{exp_name}.pt")
            print("Saved Best Model! Loss:", best_loss)
            best_epoch = epoch

        if epoch - best_epoch > patience:
            break

    print("Training completed. Best loss:", best_loss)



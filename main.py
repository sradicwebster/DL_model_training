import logging
import hydra.utils
import numpy as np
import omegaconf
import hydra
import wandb
from tqdm import tqdm

import torch.nn
import torch.optim
from torch.utils.data import DataLoader

from dataset_loader import dataset_loader
from model import MLP


def resolve_tuple(*args):
    return tuple(args)


omegaconf.OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


@hydra.main(config_path="configs", config_name="main")
def run_experiment(cfg: omegaconf.DictConfig) -> None:
    base_path = hydra.utils.get_original_cwd()
    setattr(cfg.wandb.init, "dir", base_path)

    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    train_data, test_data = dataset_loader(cfg.dataset, cfg.seed)
    input_dim = train_data.data.flatten(1, -1).shape[1]
    cfg.model.input_dim = input_dim
    if cfg.dataset.task == "classification":
        output_dim = len(train_data.classes)
    else:
        output_dim = 1
    cfg.model.output_dim = output_dim

    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb.init(**cfg.wandb.init, config=wandb_cfg) as run:

        model = MLP(input_dim,
                    output_dim,
                    cfg.model.hidden_layer_sizes,
                    getattr(torch.nn, cfg.model.activation),
                    cfg.model.squash_output,
                    cfg.model.softmax_output,
                    ).to(cfg.device)
        optimizer = getattr(torch.optim, cfg.optimiser.name)(model.parameters(), **cfg.optimiser.params)
        loss_fn = getattr(torch.nn, cfg.model.loss_fn)()
        wandb.watch(model, **cfg.wandb.watch)

        train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                                      generator=torch_generator)
        test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True,
                                     generator=torch_generator)

        for epoch in tqdm(range(cfg.epochs)):

            # train loop
            for batch, (X, y) in enumerate(train_dataloader):
                output = model(X)
                loss = loss_fn(output.squeeze(), y)
                wandb.log({"metrics/train_loss": loss})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test loop
            losses = np.zeros(len(test_dataloader))
            total_test = correct_pred = 0
            for batch, (X, y) in enumerate(test_dataloader):
                with torch.no_grad():
                    output = model(X)
                loss = loss_fn(output.squeeze(), y)
                losses[batch] = loss.detach().numpy()
                if cfg.dataset.task == "classification":
                    pred = output.argmax(1)
                    total_test += len(pred)
                    correct_pred += (pred == y).sum().item()
            wandb.log({"epoch": epoch+1, "metrics/test_loss": losses.mean()}, commit=False)
            if cfg.dataset.task == "classification":
                wandb.log({"metrics/test_accuracy": 100 * correct_pred / total_test}, commit=False)


if __name__ == "__main__":
    run_experiment()

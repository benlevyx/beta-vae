from tqdm.notebook import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FilterSampler


def train_model(model,
                dataset_train,
                dataset_eval,
                num_epochs=10,
                eval_steps=100,
                log_steps=100,
                batch_size=64,
                keep_letters=None,
                lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    if keep_letters is not None:
        sampler_train = FilterSampler(dataset_train.targets, keep_letters, dataset_train.classes)
        sampler_eval = FilterSampler(dataset_eval.targets, keep_letters, dataset_eval.classes)
    else:
        sampler_train, sampler_eval = None, None
    loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size, sampler=sampler_eval)

    model.train()

    print(f"Training for {num_epochs} epochs.")
    total_loss = 0.
    total_steps = 0

    iters_per_epoch = len(loader_train) if sampler_train is None else int(np.ceil(len(sampler_train) / batch_size))

    for i in range(num_epochs):
        print(f"Epoch {i}")
        with tqdm(total=iters_per_epoch) as pbar:
            for batch in loader_train:
                opt.zero_grad()

                pred, loss = model(batch[0])

                total_loss += loss.item()
                total_steps += 1

                loss.backward()
                opt.step()

                pbar.set_description(f'Loss: {total_loss / total_steps:.4f}')

                # if total_steps % log_steps == 0:
                #     tqdm.write(f'Iter {total_steps} -- Total loss: {total_loss / total_steps:.4f}')
                # if total_steps % eval_steps == 0:
                #     eval_loop(model, loader_eval)

                pbar.update()
        eval_loop(model, loader_eval)

    print("Training done. Doing final evaluation")
    eval_loop(model, loader_eval)

    return model


def eval_loop(model, loader_eval):
    tqdm.write("Evaluating")
    eval_loss = 0.
    eval_steps = 0
    for batch in tqdm(loader_eval):
        with torch.no_grad():
            pred, loss = model(batch[0])
            eval_loss += loss.item()
            eval_steps += 1
    tqdm.write(f'Eval loss: {eval_loss / eval_steps:.4f}')

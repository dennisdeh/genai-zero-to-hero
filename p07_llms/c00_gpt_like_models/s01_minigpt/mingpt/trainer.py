"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

Portions of this file are derived from:
https://github.com/karpathy/minGPT
Original author: Andrej Karpathy
Changes: Small modifications made to the original code to allow for setups with multiple GPUs,
added estimate_loss, batch_end_callback, early stopping
"""

import time
from collections import defaultdict
import random
import torch
from torch.utils.data.dataloader import DataLoader
from p07_llms.c00_gpt_like_models.s01_minigpt.mingpt.utils import CfgNode as CN


@torch.no_grad()
def estimate_loss(model, dataset, device, eval_iters=50):
    """
    Estimate the average loss over a specific number of iterations without modifying the state
    of the model. This function evaluates the model in evaluation mode and returns the mean
    loss computed across the specified number of iterations.

    :param model: A PyTorch model that contains a forward method or a DataParallel module for
        calculating predictions and associated loss.
    :param dataset: A dataset containing input-output pairs. Each element of the dataset
        should return a tuple of (features, labels).
    :param device: The computational device (e.g., "cpu" or "cuda") on which the evaluation
        should occur.
    :param eval_iters: The number of iterations to evaluate the model for computing the
        average loss. Defaults to 50.
    :return: The mean loss computed over the specified number of evaluation iterations.
    :rtype: float
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # Sample a random batch from the dataset
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        logits, loss = model(x, y)
        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def batch_end_callback(trainer, dataset_val, device=None):
    """
    To be triggered at the end of every batch during training to log progress and validation
    loss at specified intervals. This callback function ensures that information about
    training and validation is printed periodically.

    :param trainer: Trainer instance that contains the training model and loss. Must have
        attributes ``model`` and ``loss``.
    :type trainer: Any

    :param dataset_val: Validation dataset used to calculate validation loss. This must
        not be ``None``.
    :type dataset_val: Any

    :param device: The device (e.g., CPU or GPU) where validation computations are performed.
        This must not be ``None``.
    :type device: Any

    :return: val_loss
    """
    assert hasattr(trainer, "model")
    assert hasattr(trainer, "loss")
    assert dataset_val is not None
    assert device is not None

    str_print = ""
    val_loss = None
    if trainer.iter_num % 25 == 0:
        str_print = f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"

    if trainer.iter_num % 100 == 0:
        val_loss = estimate_loss(
            model=trainer.model, dataset=dataset_val, device=device
        )
        str_print += f" -- validation loss {val_loss:.5f}"

    if str_print != "":
        print(str_print)
    return val_loss


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloader parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # early stopping parameters
        C.early_stopping_rounds = 300
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        if hasattr(self.model, "module"):
            print(f"Running on {torch.cuda.device_count()} {self.device} devices")
        else:
            print(f"Running on a single {self.device} device ")
        self.model = self.model.to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        # variables for early stopping
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        results = []
        for callback in self.callbacks.get(onevent, []):
            results.append(callback(self))
        return results

    def run(self):
        model, config = self.model, self.config

        # setup the optimiser
        # handle the case where the model is wrapped in DataParallel
        raw_model = model.module if hasattr(model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=None
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)
            # collapse all losses if they are scattered on multiple gpus
            self.loss = self.loss.mean()

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            res = self.trigger_callbacks("on_batch_end")
            # the callback might return the validation loss and in this case the early stopping counter is updated
            val_loss = next((r for r in res if r is not None), None)
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
            if (
                config.early_stopping_rounds is not None
                and self.early_stopping_counter >= config.early_stopping_rounds
            ):
                print(f"Early stopping triggered after {self.iter_num} iterations.")
                break

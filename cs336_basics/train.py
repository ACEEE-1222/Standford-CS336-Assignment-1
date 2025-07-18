import argparse
import os
from typing import IO, BinaryIO, Callable, Iterable, Optional, Union
import torch
from torch import nn
from einops import einsum, rearrange
import math
import numpy.typing as npt
import numpy as np
from tqdm import tqdm, trange
import transformer

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss over a batch.

    Args:
        inputs (Tensor): Float tensor of shape (batch_size, vocab_size), representing unnormalized logits.
        targets (Tensor): Int tensor of shape (batch_size,), representing the indices of correct classes.

    Returns:
        Tensor: A scalar tensor containing the average cross-entropy loss over the batch.
    """
    batch_size = inputs.shape[0]
    o_max = torch.max(inputs, dim=-1, keepdim=True).values
    o = inputs - o_max
    target_logits = o[torch.arange(batch_size), targets]
    logsumexp = torch.log(torch.sum(torch.exp(o), dim=-1))
    loss = -target_logits + logsumexp
    loss_ave = loss.mean(dim=0)
    return loss_ave

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)


    def step(self, closure: Callable | None = None):  
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            grad = p.grad.data

            t = state.get("t", 1) # Get iteration number from the state, or initial value.
            m = state.get("m", torch.zeros_like(grad)) 
            v = state.get("v", torch.zeros_like(grad)) 

             # Get the gradient of loss with respect to p.
            m = beta1 * m + (1-beta1) * grad
            v = beta2 * v + (1-beta2) * grad**2
            lr_t = lr * (1-beta2**t)**0.5/(1-beta1**t)
            p.data = p.data - lr_t * m / (v**0.5+eps)
            p.data = p.data - lr * weight_decay * p.data
            state["t"] = t + 1 # Increment iteration number.
            state["m"] = m
            state["v"] = v
        return loss

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    lr = 0
    if t < warmup_iters:
        lr = t / warmup_iters * lr_max
    elif t < cosine_cycle_iters:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
    else:
        lr = lr_min

    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 
    l2_norm = 0.0
    for g in grads:
        l2_norm += g.sum()
    l2_norm = torch.sqrt(l2_norm)

    clip_coef = min(1, max_l2_norm / (l2_norm + 1e-6))
    for g in grads:
        g *= clip_coef
    
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    max_start = len(dataset) - context_length - 1
    if max_start <= 0:
        raise ValueError("Dataset too short for the given context length.")

    starts = np.random.randint(0, max_start + 1, size=batch_size)

    x_batch = []
    y_batch = []
    for s in starts:
        seq = dataset[s : s + context_length + 1]  
        x_batch.append(seq[:-1]) 
        y_batch.append(seq[1:]) 


    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    
    return x, y



def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets using memory mapping
    train_data = np.load(args.train_data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r')

    model = transformer.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load from checkpoint if available
    start_iter = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    progress_bar = trange(args.start_iter, args.total_iters, desc="Training")

    for t in progress_bar:
        # Learning rate schedule
        lr = lr_cosine_schedule(t, args.lr, args.min_lr, args.warmup_iters, args.total_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train step
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging to progress bar
        if t % args.log_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=lr)

        # Evaluation
        if t % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                val_logits = model(x_val)
                val_loss = cross_entropy(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
                tqdm.write(f"[Eval @ Iter {t}] Val loss {val_loss.item():.4f}")

        # Checkpoint saving
        if args.checkpoint_path and t % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, t, args.checkpoint_path)
            tqdm.write(f"[Checkpoint @ Iter {t}] Saved to {args.checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint.pt")

    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--context_length', type=int, default=32)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--theta', type=float, default=None)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--total_iters', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=500)

    args = parser.parse_args()
    train(args)

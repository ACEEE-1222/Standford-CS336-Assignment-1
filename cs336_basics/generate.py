import torch
import torch.nn.functional as F

@torch.no_grad()
def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """
    Generate text from the model given a prompt.

    Args:
        model: Trained TransformerLM model.
        tokenizer: Tokenizer with encode() and decode().
        prompt (str): The starting text to condition on.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for softmax sampling.
        top_p (float): Nucleus sampling threshold (0 < p <= 1).
        device: Device to run on.

    Returns:
        Generated string including the prompt.
    """
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)  # shape: (1, seq_len, vocab_size)
        logits = logits[:, -1, :]  # last token's logits: (1, vocab_size)

        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)  # shape: (1, vocab_size)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
            break

    return tokenizer.decode(input_ids[0].tolist())

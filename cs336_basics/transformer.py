import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Constructs a linear transformation module without bias.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input: y = x @ W.T

        Args:
            x (torch.Tensor): Input tensor of shape (... , in_features)

        Returns:
            torch.Tensor: Output tensor of shape (... , out_features)
        """

        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors (d_model).
            device (torch.device | None): Device to store parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.d_model), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)
        

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Lookup embedding vectors for the given token IDs.

        Args:
            token_ids (torch.LongTensor): LongTensor of shape (batch_size, sequence_length),
                                      containing token indices.

        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, sequence_length, embedding_dim).
        """

        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): Epsilon value for numerical stability.
            device (torch.device | None): Device to store parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Root Mean Square Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype= torch.float32)
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps)**0.5
        out = x / rms * self.weight

        return out.to(dtype=in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int ,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Dimension of the feed-forward layer.
            device (torch.device | None): Device to store parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def _glu(self, x: torch.Tensor) -> torch.Tensor:
        return self._silu(self.w1(x)) * self.w3(x)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._glu(x))
        

        
class ROPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        """
        Construct the Rotary Positional Embedding (RoPE) module.

        Args:
            theta (float): RoPE base frequency hyperparameter.
            d_k (int): Dimension of query/key vectors (must be even).
            max_seq_len (int): Maximum sequence length to support.
            device (torch.device | None): Device to store the cos/sin buffers.
        """
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
            freqs_d = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
            pos_i = torch.arange(max_seq_len, device=device).float()
            freqs = einsum(freqs_d, pos_i, "d_half, max_seq_len -> max_seq_len d_half")

            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                
        token_positions: torch.Tensor  
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) indicating token positions.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, shape same as x.
        """
        x_odd = x[...,1::2]
        x_even = x[...,::2]

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        out1 = cos * x_even - sin * x_odd # (..., seq_len, d_k//2)
        out2 = sin * x_even + cos * x_odd
        out = torch.stack([out1, out2], dim=-1).flatten(-2)

        return out


def softmax(x: torch.tensor,
            dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / sum_exp

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes the scaled dot-product attention.

    Args:
        query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Tensor of shape (batch_size, ..., seq_len_k, d_k)
        value: Tensor of shape (batch_size, ..., seq_len_k, d_v)
        mask: Optional boolean tensor of shape (seq_len_q, seq_len_k),
              where True indicates positions to attend, False means masked.

    Returns:
        output: Tensor of shape (batch_size, ..., seq_len_q, d_v)
    """
    d_k = query.shape[-1]
    attention = einsum(query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / d_k ** 0.5
    if mask is not None:
        attention = attention.masked_fill(~mask, float('-inf'))
    attention = softmax(attention, dim=-1)
    out = einsum(attention, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                theta: float | None = None,
                max_seq_len: int | None = None,
                ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads*self.d_k)
        self.k_proj = Linear(d_model, num_heads*self.d_k)
        self.v_proj = Linear(d_model, num_heads*self.d_v)
        self.output_proj = Linear(num_heads*self.d_v, d_model)

        if theta is not None and max_seq_len is not None:
            self.rope = ROPE(theta, self.d_k, max_seq_len)

    def forward(self, x: torch.Tensor, 
                mask: torch.Tensor | None = None,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *b, seq_len, d_model = x.shape
        x_q = self.q_proj(x)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x)
        
        x_q = rearrange(x_q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                        num_heads=self.num_heads, d_k=self.d_k)
        x_k = rearrange(x_k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                        num_heads=self.num_heads, d_k=self.d_k)
        x_v = rearrange(x_v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", 
                        num_heads=self.num_heads, d_v=self.d_v)
        
        # ROPE
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            for _ in range(len(b)):
                token_positions = token_positions.unsqueeze(0)  
            x_q = self.rope(x_q, token_positions)
            x_k = self.rope(x_k, token_positions)
        
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            for _ in range(len(b)):
                mask = mask.unsqueeze(0)  
        else:
            for _ in range(len(b)):
                mask = mask.unsqueeze(0) 
        
        out = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        out = rearrange(out, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", 
                        num_heads=self.num_heads, d_v=self.d_v)
        out = self.output_proj(out)

        return out

        

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float | None = None,
                max_seq_len: int | None = None,):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
        if theta is not None and max_seq_len is not None:
            # self.theta = theta 
            # self.max_seq_len = max_seq_len
            self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        else:
            self.attn = MultiHeadSelfAttention(d_model, num_heads)
            
    
    def forward(self, 
            x: torch.Tensor, 
            mask: torch.Tensor | None = None, 
            token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int, num_heads: int, d_ff: int,
                 theta: float | None = None,):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.softmax = softmax

    def forward(self, 
            inputs: torch.Tensor) -> torch.Tensor:
        
        x = self.token_embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        # out = self.softmax(self.lm_head(self.ln_final(x)), dim=-1)
        out = self.lm_head(self.ln_final(x))
        return out





        
    




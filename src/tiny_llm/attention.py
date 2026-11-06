import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if not scale:
        scale = 1 / mx.sqrt(query.shape[-1])
    score = (query @ mx.swapaxes(key,-2,-1)) * scale
    if mask is not None:
        score += mask
    score = softmax(score, -1)
    score = score @ value
    return score


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        # E = H x D
        self.model_dim = hidden_size
        # H
        self.num_heads = num_heads
        # D
        self.d_k = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        n = query.shape[0]
        l = query.shape[1]
        # project from N L E -> N L H D
        q = mx.reshape(linear(query, self.wq), [n, l, self.num_heads, self.d_k])
        k = mx.reshape(linear(key, self.wk), [n, l, self.num_heads, self.d_k])
        v = mx.reshape(linear(value, self.wv), [n, l, self.num_heads, self.d_k])
        # Transpose to N H L D
        q = mx.swapaxes(q, 1, 2)
        k = mx.swapaxes(k, 1, 2)
        v = mx.swapaxes(v, 1, 2)
        mat = scaled_dot_product_attention_simple(q,k,v,None,mask)
        # back to N L H D
        mat = mx.swapaxes(mat, 1, 2)
        # N L E
        mat = mx.reshape(mat, [n, l, self.model_dim])
        # returns N L E
        return linear(mat, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass

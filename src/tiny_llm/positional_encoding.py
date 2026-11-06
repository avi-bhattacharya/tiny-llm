import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # dims = D from attention
        self.D = dims
        self.M = dims // 2
        # L ??
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.angular_rate = []
        # mx.arange(0,self.M,1)
        if traditional:

        

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        pass

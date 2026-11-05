import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.D = dims
        self.M = dims // 2
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.angular_rate = []
        # mx.arange(0,self.M,1)
        for i in range(self.M):
            num = base ** (-i / (self.M))
            self.angular_rate.append(num)
        

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        pass

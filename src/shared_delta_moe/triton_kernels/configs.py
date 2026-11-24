"""Autotune configurations shared by the grouped-GEMM kernels."""

import triton


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in (
            (16, 64, 128, 4, 4),
            (16, 128, 64, 4, 4),
            (32, 64, 128, 4, 4),
            (32, 128, 64, 4, 4),
            (64, 128, 64, 4, 3),
            (64, 128, 64, 8, 4),
            (128, 128, 64, 8, 3),
            (256, 128, 64, 8, 3),
            (64, 64, 64, 4, 4),
            (128, 64, 64, 4, 4),
            (64, 128, 32, 4, 5),
            (128, 128, 32, 8, 4),
            (128, 256, 32, 8, 4),
            (256, 128, 32, 8, 4),
        )
    ]

"""DARE and BitDelta ``_unpack`` (block scatter; one-pass sign select) equal the earlier masked_scatter /
repeat_interleave formulas bit for bit, for element and block DARE and every BitDelta scale shape. CPU."""

import torch

from shared_delta_moe import BitDeltaMoE, DareDeltaMoE, DeltaMoE, random_experts
from shared_delta_moe.models.bitdelta import unpack_signs
from shared_delta_moe.ops import unpack_bits


def test_unpack_matches_reference():
    torch.manual_seed(0)
    full = random_experts(64, 96, 4, torch.float32, "cpu")
    m = DeltaMoE.from_experts(full[0], full[1:], torch.randn(4, 64), top_k=1, num_zero_experts=1)
    for block in (1, 8, 32):
        d = DareDeltaMoE.from_delta_moe(m, keep=0.3, block=block, seed=1)
        for n in ("gate", "up", "down"):
            _, R, C = d.shapes[n]
            v = getattr(d, f"v_{n}")
            for e in range(3):
                keep = unpack_bits(d._bits(n)[e])[:, : C // block].bool().repeat_interleave(block, 1)
                ref = v.new_zeros(R, C).masked_scatter(keep, v[int(getattr(d, f"p_{n}")[e, 0]) :])
                assert torch.equal(ref, d._unpack(n, e)), (block, n, e)
    for scale in ("matrix", "row", 8, 32):
        b = BitDeltaMoE.from_delta_moe(m, scale=scale)
        for n in ("gate", "up", "down"):
            s, C = getattr(b, f"s_{n}"), b.shapes[n][2]
            for e in range(3):
                ref = s[e].repeat_interleave(b.sblock[n], 1)[:, :C] * unpack_signs(getattr(b, f"b_{n}")[e], s.dtype)
                assert torch.equal(ref, b._unpack(n, e)), (scale, n, e)


if __name__ == "__main__":
    test_unpack_matches_reference()
    print("ok")

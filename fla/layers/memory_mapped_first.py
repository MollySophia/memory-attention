"""Fresh mapped lookup for the first pipeline group only."""
import triton
import triton.language as tl


@triton.jit
def _gather(table, ids, output, TOKENS: tl.constexpr, WIDTH: tl.constexpr,
            STRIDE: tl.constexpr, VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.cdiv(WIDTH, BLOCK)
    for tile in range(tl.program_id(0), TOKENS * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = (tile % tiles) * BLOCK + tl.arange(0, BLOCK)
        token = tl.load(ids + row)
        value = tl.load(table + token * STRIDE + column,
                        mask=(column < WIDTH) & (token >= 0) & (token < VOCAB), other=0)
        tl.store(output + row * WIDTH + column, value, mask=column < WIDTH)


def mapped_first_group(table, ids, output, count):
    width = count * table.shape[2]
    grid = min(64, ids.numel() * triton.cdiv(width, 512))
    _gather[(grid,)](table, ids, output, ids.numel(), width,
                     table.shape[1] * table.shape[2], table.shape[0], 512, num_warps=4)

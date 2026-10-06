"""One module per graph node (SPEC.md §5.2).

Each node is `run(ctx, state, config) -> dict` of state updates. No node waits
for learning: they only call `ctx.learning.emit(...)` (SPEC.md §8.6).
"""

"""Behavioral scheduling seam used by the transformed GR00T runner."""


def advance_chunk_index(index: int, rows_due: int, horizon: int) -> tuple[int, bool]:
    """Select the newest due row; return its index and whether to publish it."""
    if rows_due < 0 or horizon <= 0:
        raise ValueError("invalid action scheduling state")
    if rows_due == 0:
        return index, False
    return min(index + rows_due - 1, horizon - 1), True

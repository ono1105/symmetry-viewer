from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AffineOperation:
    """Fractional affine operation x' = W @ x + t."""

    W: np.ndarray
    t: np.ndarray


def normalize_translation(t: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """Normalize a fractional translation into [0, 1) component-wise."""
    normalized = np.asarray(t, dtype=float) % 1.0
    normalized[np.isclose(normalized, 0.0, atol=atol)] = 0.0
    normalized[np.isclose(normalized, 1.0, atol=atol)] = 0.0
    return normalized


def compose_affine_operations(
    first: tuple[np.ndarray, np.ndarray] | AffineOperation,
    second: tuple[np.ndarray, np.ndarray] | AffineOperation,
    *,
    normalize: bool = True,
) -> AffineOperation:
    """Compose two operations, applying first then second.

    For A=(W_A,t_A) followed by B=(W_B,t_B):
      W = W_B @ W_A
      t = W_B @ t_A + t_B
    """
    W_first, t_first = operation_components(first)
    W_second, t_second = operation_components(second)
    W = W_second @ W_first
    t = W_second @ t_first + t_second
    if normalize:
        t = normalize_translation(t)
    return AffineOperation(W=W, t=t)


def compose_operation_sequence(
    operations: Sequence[tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any],
    *,
    normalize: bool = True,
) -> AffineOperation:
    """Compose a sequence in application order: operations[0], then operations[1], ..."""
    result = AffineOperation(W=np.eye(3), t=np.zeros(3))
    for operation in operations:
        result = compose_affine_operations(result, operation, normalize=normalize)
    return result


def operations_equivalent(
    left: tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any,
    right: tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any,
    *,
    atol: float = 1e-8,
) -> bool:
    """Return True when two fractional affine operations differ only by a lattice translation."""
    W_left, t_left = operation_components(left)
    W_right, t_right = operation_components(right)
    if not np.allclose(W_left, W_right, atol=atol):
        return False
    return translations_equivalent(t_left, t_right, atol=atol)


def translations_equivalent(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-8) -> bool:
    """Compare fractional translations modulo integer lattice vectors."""
    delta = normalize_translation(np.asarray(left, dtype=float) - np.asarray(right, dtype=float), atol=atol)
    centered = delta - np.round(delta)
    return bool(np.allclose(centered, np.zeros(3), atol=atol))


def find_matching_operation_index(
    W: np.ndarray,
    t: np.ndarray,
    operations: Iterable[dict[str, Any] | Any],
    *,
    atol: float = 1e-8,
) -> int | None:
    """Find the index of an existing operation matching (W,t), modulo lattice translations."""
    target = AffineOperation(W=np.asarray(W), t=np.asarray(t))
    for operation in operations:
        try:
            if operations_equivalent(target, operation, atol=atol):
                return operation_index(operation)
        except (KeyError, AttributeError, ValueError, TypeError):
            continue
    return None


def find_operation_sequence_bfs(
    target: tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any,
    generators: Sequence[dict[str, Any] | Any],
    operations: Iterable[dict[str, Any] | Any],
    *,
    max_depth: int = 4,
    atol: float = 1e-8,
) -> tuple[int, ...] | None:
    """Find a short generator sequence that composes to target.

    The returned indices are in application order.  Identity is represented by
    an empty sequence when the target itself is equivalent to identity.
    """
    if max_depth < 0:
        return None
    identity = AffineOperation(W=np.eye(3), t=np.zeros(3))
    if operations_equivalent(identity, target, atol=atol):
        return ()

    known_operations = tuple(operations)
    generator_items = [
        (operation_index(generator), generator)
        for generator in generators
    ]
    queue = deque([(identity, ())])
    seen = {("raw", operation_key(identity, atol=atol))}
    while queue:
        current, sequence = queue.popleft()
        if len(sequence) >= max_depth:
            continue
        for generator_index, generator in generator_items:
            candidate = compose_affine_operations(current, generator)
            matching_index = find_matching_operation_index(candidate.W, candidate.t, known_operations, atol=atol)
            key = ("op", matching_index) if matching_index is not None else ("raw", operation_key(candidate, atol=atol))
            if key in seen:
                continue
            next_sequence = sequence + (generator_index,)
            if operations_equivalent(candidate, target, atol=atol):
                return next_sequence
            seen.add(key)
            queue.append((candidate, next_sequence))
    return None


def operation_key(
    operation: tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any,
    *,
    atol: float = 1e-8,
) -> tuple:
    W, t = operation_components(operation)
    if atol <= 0:
        W_key = tuple(np.asarray(W).reshape(-1).tolist())
        t_key = tuple(normalize_translation(t).tolist())
        return W_key, t_key
    W_key = tuple(np.rint(np.asarray(W, dtype=float).reshape(-1) / atol).astype(int).tolist())
    t_key = tuple(np.rint(normalize_translation(t, atol=atol) / atol).astype(int).tolist())
    return W_key, t_key


def operation_components(
    operation: tuple[np.ndarray, np.ndarray] | AffineOperation | dict[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(operation, AffineOperation):
        return np.asarray(operation.W), np.asarray(operation.t, dtype=float)
    if isinstance(operation, tuple) and len(operation) == 2:
        W, t = operation
        return np.asarray(W), np.asarray(t, dtype=float)
    if isinstance(operation, dict):
        W = operation.get("W", operation.get("matrix_frac", operation.get("rotation")))
        t = operation.get("t", operation.get("translation_frac", operation.get("translation")))
        if W is None or t is None:
            raise KeyError("operation must contain W/t or matrix_frac/translation_frac")
        return np.asarray(W), np.asarray(t, dtype=float)
    W = getattr(operation, "W", getattr(operation, "matrix_frac", getattr(operation, "rotation", None)))
    t = getattr(operation, "t", getattr(operation, "translation_frac", getattr(operation, "translation", None)))
    if W is None or t is None:
        raise AttributeError("operation must expose W/t or matrix_frac/translation_frac")
    return np.asarray(W), np.asarray(t, dtype=float)


def operation_index(operation: dict[str, Any] | Any) -> int:
    if isinstance(operation, dict):
        return int(operation["index"])
    return int(getattr(operation, "index"))

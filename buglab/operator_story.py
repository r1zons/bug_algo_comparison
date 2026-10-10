"""Reproducible teaching examples and records for GA genetic operations."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from .metrics import OutcomeMetrics


@dataclass(frozen=True)
class SyntheticOperation:
    """A small, deterministic one-point crossover and mutation example."""

    parent_a: np.ndarray
    parent_b: np.ndarray
    cut: int
    child_before_mutation: np.ndarray
    mutated_indices: tuple[int, ...]
    mutation_deltas: np.ndarray
    child: np.ndarray


@dataclass(frozen=True)
class GeneticOperation:
    """One candidate's provenance across two aligned ask/tell batches."""

    seed: int
    generation: int
    operation_id: int
    candidate_index: int
    kind: str
    parent_a_id: int
    parent_b_id: int | None
    parent_a: np.ndarray
    parent_b: np.ndarray | None
    parent_a_score: float
    parent_b_score: float | None
    cut: int | None
    child_before_mutation: np.ndarray
    mutated_indices: tuple[int, ...]
    mutation_deltas: np.ndarray
    child: np.ndarray
    child_score: float | None = None
    child_metrics: OutcomeMetrics | None = None

    @property
    def better_parent_score(self) -> float:
        if self.parent_b_score is None:
            return self.parent_a_score
        return max(self.parent_a_score, self.parent_b_score)


@dataclass(frozen=True)
class OperationSelection:
    """One deterministic selection per teaching category, or ``None``."""

    improvement: GeneticOperation | None
    neutral: GeneticOperation | None
    degradation: GeneticOperation | None


def synthetic_operation() -> SyntheticOperation:
    """Return the canonical eight-gene example, with radians stored internally."""
    parent_a = np.deg2rad(np.array([-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0]))
    parent_b = np.deg2rad(np.array([20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0]))
    cut = 4
    before = np.concatenate((parent_a[:cut], parent_b[cut:]))
    indices = (2, 6)
    deltas = np.deg2rad(np.array([7.0, -8.0]))
    child = before.copy()
    child[np.asarray(indices)] += deltas
    return SyntheticOperation(parent_a, parent_b, cut, before, indices, deltas, child)


def reconstruct_child(operation: GeneticOperation | SyntheticOperation) -> np.ndarray:
    """Reconstruct a final child exactly from its pre-mutation values and deltas."""
    child = operation.child_before_mutation.copy()
    child[np.asarray(operation.mutated_indices, dtype=int)] += operation.mutation_deltas
    return child


def with_outcome(operation: GeneticOperation, metrics: OutcomeMetrics) -> GeneticOperation:
    """Attach experiment-owned physical outcomes without exposing them to the GA."""
    return replace(operation, child_metrics=metrics)


def classification_tolerance(
    operation: GeneticOperation,
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 0.25,
) -> float:
    """Return the documented near-neutral band around the better parent score."""
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Classification tolerances must be non-negative")
    return max(absolute_tolerance, relative_tolerance * max(1.0, abs(operation.better_parent_score)))


def classify_operation(
    operation: GeneticOperation,
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 0.25,
) -> str:
    """Classify offspring relative to its better parent; elites remain explicit."""
    if operation.kind == "elite":
        return "elite"
    if operation.child_score is None:
        return "pending"
    better = operation.better_parent_score
    worse = operation.parent_a_score if operation.parent_b_score is None else min(
        operation.parent_a_score, operation.parent_b_score
    )
    delta = operation.child_score - better
    tolerance = classification_tolerance(
        operation,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if delta > tolerance:
        return "improvement"
    if abs(delta) <= tolerance:
        return "neutral"
    if operation.child_score < worse - tolerance:
        return "degradation"
    return "mixed"


def select_operation_examples(
    operations: list[GeneticOperation],
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 0.25,
) -> OperationSelection:
    """Select the first recorded event in each category, honestly preserving gaps."""
    selected: dict[str, GeneticOperation | None] = {
        "improvement": None,
        "neutral": None,
        "degradation": None,
    }
    for operation in operations:
        label = classify_operation(
            operation,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if label in selected and selected[label] is None:
            selected[label] = operation
        if all(value is not None for value in selected.values()):
            break
    return OperationSelection(**selected)


def save_operation_log(operations: list[GeneticOperation], path: Path) -> None:
    """Write a replayable JSON log; arrays remain radians and metrics stay explicit."""
    def serializable(operation: GeneticOperation) -> dict:
        record = asdict(operation)
        for key in ("parent_a", "parent_b", "child_before_mutation", "mutation_deltas", "child"):
            value = record[key]
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
        record["mutated_indices"] = list(operation.mutated_indices)
        record["classification"] = classify_operation(operation)
        return record

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump([serializable(operation) for operation in operations], file, indent=2)


def load_operation_log(path: Path) -> list[GeneticOperation]:
    """Load a JSON operation log for deterministic replay."""
    with path.open() as file:
        records = json.load(file)
    operations = []
    for record in records:
        record.pop("classification", None)
        for key in ("parent_a", "child_before_mutation", "mutation_deltas", "child"):
            record[key] = np.asarray(record[key], dtype=float)
        if record["parent_b"] is not None:
            record["parent_b"] = np.asarray(record["parent_b"], dtype=float)
        record["mutated_indices"] = tuple(record["mutated_indices"])
        if record["child_metrics"] is not None:
            record["child_metrics"] = OutcomeMetrics(**record["child_metrics"])
        operations.append(GeneticOperation(**record))
    return operations

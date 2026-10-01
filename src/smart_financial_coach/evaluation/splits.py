"""Splits: named sets of example IDs plus validation folds, built by small composable splitters.

Every splitter is a pure function of the examples and a seed, works on example IDs and group
columns (not on transactions specifically), and returns sorted IDs, so a split is reproducible and
hashable. `leak_errors` runs on every split before any model is fitted.

    test_known = stratified_sample(train_users, frac=0.2, by="category", seed=0, id_column="id")
    folds = group_kfold(rest, group="merchant_id", k=5, seed=0, id_column="id", seen_frac=0.1)
    splits = Splits(sets={"train": ..., "test_known": test_known}, folds=folds)
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

Ids = npt.NDArray[np.str_]
TRAIN = "train"
UNSEEN = "unseen"  # fold held-out set: groups the fold model never saw
SEEN = "seen"  # fold held-out set: rows of groups the fold model did see


def ids(values: pd.Series | list[str] | npt.NDArray[np.str_]) -> Ids:
    return np.sort(np.asarray(values, dtype=str))


def _hash(values: Ids) -> str:
    return hashlib.sha256("\x1f".join(values.tolist()).encode()).hexdigest()


@dataclass(frozen=True)
class Fold:
    train: Ids
    held_out: Mapping[str, Ids]  # named held-out sets, e.g. {"unseen": ..., "seen": ...}


@dataclass(frozen=True)
class Splits:
    sets: Mapping[str, Ids]  # must include "train"; test sets are scored only by `finalize`
    folds: tuple[Fold, ...] = field(default=())

    def hashes(self) -> dict[str, str]:
        out = {name: _hash(values) for name, values in sorted(self.sets.items())}
        for i, fold in enumerate(self.folds):
            out[f"fold{i}.train"] = _hash(fold.train)
            for name, values in sorted(fold.held_out.items()):
                out[f"fold{i}.{name}"] = _hash(values)
        return out

    def hash(self) -> str:
        """One hash for the whole split: runs are comparable only if theirs are equal."""
        joined = "\x1e".join(f"{k}={v}" for k, v in self.hashes().items())
        return hashlib.sha256(joined.encode()).hexdigest()


def stratified_sample(
    frame: pd.DataFrame, *, frac: float, by: str, seed: int, id_column: str
) -> Ids:
    """A seeded `frac` of each stratum's rows (rounded), independent of row order."""
    rng = np.random.default_rng(seed)
    picked: list[Ids] = []
    for _, rows in sorted(frame.groupby(by, sort=True), key=lambda kv: str(kv[0])):
        stratum = ids(rows[id_column])
        n = round(frac * len(stratum))
        picked.append(rng.choice(stratum, size=n, replace=False) if n else stratum[:0])
    return ids(np.concatenate(picked)) if picked else ids([])


def group_kfold(
    frame: pd.DataFrame,
    *,
    group: str,
    k: int,
    seed: int,
    id_column: str,
    stratify: str | None = None,
    eligible: set[str] | None = None,
    seen_frac: float = 0.0,
) -> tuple[Fold, ...]:
    """K folds that each hold out whole groups (e.g. merchants) the fold model never sees.

    - Only `eligible` groups are ever held out (default: all); the rest stay in every fold's
      training rows. Each eligible group is held out in exactly one fold.
    - With `stratify`, groups are dealt to folds within each value of their majority stratum,
      so every fold gets groups from most strata.
    - With `seen_frac`, each fold also holds out a seeded share of its remaining rows, which
      measures the model on groups it has seen.
    """
    if k < 2:
        raise ValueError("group_kfold needs k >= 2")
    rng = np.random.default_rng(seed)
    groups = frame[group].astype(str)
    candidates = sorted(set(groups) if eligible is None else set(groups) & set(eligible))
    if len(candidates) < k:
        raise ValueError(f"{len(candidates)} eligible groups can't fill {k} folds")
    if stratify is None:
        strata = dict.fromkeys(candidates, "")
    else:
        majority = frame.groupby(groups)[stratify].agg(lambda s: s.value_counts().index[0])
        strata = {g: str(majority[g]) for g in candidates}

    assignment: dict[str, int] = {}
    offset = 0
    for stratum in sorted(set(strata.values())):
        members = np.asarray(sorted(g for g in candidates if strata[g] == stratum), dtype=str)
        for i, g in enumerate(rng.permutation(members)):
            assignment[str(g)] = (offset + i) % k
        offset += len(members)

    fold_of = groups.map(assignment)
    folds = []
    for f in range(k):
        held = frame[fold_of == f]
        rest = frame[fold_of != f]
        held_out = {UNSEEN: ids(held[id_column])}
        train = ids(rest[id_column])
        if seen_frac > 0:
            n = round(seen_frac * len(train))
            seen = ids(rng.choice(train, size=n, replace=False))
            held_out[SEEN] = seen
            train = np.setdiff1d(train, seen, assume_unique=True)
        folds.append(Fold(train=train, held_out=held_out))
    return tuple(folds)


def leak_errors(
    splits: Splits,
    frame: pd.DataFrame,
    *,
    id_column: str,
    group: str | None = None,
    never_held_out: set[str] | None = None,
) -> list[str]:
    """Generic leak checks; tasks add their own.

    - `train` shares no ID with any other set.
    - Each fold's IDs come from `train`; its training and held-out sets share no ID.
    - With `group`: no group held out as unseen appears in that fold's training rows, and no
      group in `never_held_out` is held out as unseen.
    """
    errors = []
    if TRAIN not in splits.sets:
        return [f"splits have no {TRAIN!r} set"]
    train = splits.sets[TRAIN]
    for name, values in splits.sets.items():
        if name != TRAIN and (n := np.intersect1d(train, values).size):
            errors.append(f"{n} IDs in both {TRAIN} and {name}")
    group_of = frame.set_index(id_column)[group].astype(str) if group else None
    for i, fold in enumerate(splits.folds):
        used = [fold.train, *fold.held_out.values()]
        if (n := np.setdiff1d(np.concatenate(used), train).size) > 0:
            errors.append(f"fold {i}: {n} IDs not in {TRAIN}")
        for name, values in fold.held_out.items():
            if n := np.intersect1d(fold.train, values).size:
                errors.append(f"fold {i}: {n} IDs both trained on and held out as {name}")
        if group_of is not None and UNSEEN in fold.held_out:
            unseen = set(group_of.loc[fold.held_out[UNSEEN].tolist()])
            if shared := unseen & set(group_of.loc[fold.train.tolist()]):
                errors.append(f"fold {i}: unseen groups in training rows: {sorted(shared)[:5]}")
            if never_held_out and (bad := unseen & never_held_out):
                errors.append(f"fold {i}: protected groups held out: {sorted(bad)[:5]}")
    return errors

"""Splitters and leak checks: each invariant, and each check catching a constructed leak."""

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.evaluation.splits import (
    SEEN,
    TRAIN,
    UNSEEN,
    Fold,
    Splits,
    group_kfold,
    ids,
    leak_errors,
    stratified_sample,
)


def frame(groups: int = 20, rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": f"r{g:02d}-{i:02d}", "group": f"g{g:02d}", "label": "ab"[g % 2]}
            for g in range(groups)
            for i in range(rows)
        ]
    )


def test_stratified_sample_takes_share_of_each_stratum() -> None:
    f = frame()
    sample = stratified_sample(f, frac=0.2, by="label", seed=0, id_column="id")
    labels = f.set_index("id")["label"].loc[sample.tolist()]

    assert len(sample) == 40
    assert labels.value_counts().to_dict() == {"a": 20, "b": 20}


def test_stratified_sample_ignores_row_order() -> None:
    f = frame()
    a = stratified_sample(f, frac=0.2, by="label", seed=3, id_column="id")
    b = stratified_sample(f.iloc[::-1], frac=0.2, by="label", seed=3, id_column="id")
    c = stratified_sample(f, frac=0.2, by="label", seed=4, id_column="id")

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_group_kfold_holds_out_each_eligible_group_once() -> None:
    f = frame()
    protected = {"g00", "g01"}
    folds = group_kfold(
        f,
        group="group",
        k=4,
        seed=0,
        id_column="id",
        stratify="label",
        eligible=set(f["group"]) - protected,
        seen_frac=0.1,
    )
    group_of = f.set_index("id")["group"]
    held = [set(group_of.loc[fold.held_out[UNSEEN].tolist()]) for fold in folds]

    assert sorted(g for h in held for g in h) == sorted(set(f["group"]) - protected)
    assert all(not h & protected for h in held)
    for fold, unseen in zip(folds, held, strict=True):
        assert not unseen & set(group_of.loc[fold.train.tolist()])
        assert protected <= set(group_of.loc[fold.train.tolist()])
        assert not set(fold.train) & set(fold.held_out[SEEN])
        labels = f.set_index("id")["label"]
        assert {"a", "b"} <= set(labels.loc[fold.held_out[UNSEEN].tolist()])


def test_group_kfold_is_deterministic() -> None:
    a = group_kfold(frame(), group="group", k=3, seed=1, id_column="id", seen_frac=0.1)
    b = group_kfold(frame(), group="group", k=3, seed=1, id_column="id", seen_frac=0.1)

    assert Splits({TRAIN: ids([])}, a).hash() == Splits({TRAIN: ids([])}, b).hash()


def test_group_kfold_needs_enough_groups() -> None:
    with pytest.raises(ValueError, match="can't fill"):
        group_kfold(frame(groups=2), group="group", k=3, seed=0, id_column="id")


def test_split_hash_changes_with_any_set() -> None:
    a = Splits({TRAIN: ids(["1", "2"]), "test": ids(["3"])})
    b = Splits({TRAIN: ids(["1", "3"]), "test": ids(["2"])})

    assert a.hash() == Splits({TRAIN: ids(["2", "1"]), "test": ids(["3"])}).hash()
    assert a.hash() != b.hash()


def _clean() -> tuple[Splits, pd.DataFrame]:
    f = frame(groups=4, rows=2)  # r00-00, r00-01, r01-00 ... groups g00-g03
    fold = Fold(
        train=ids(["r00-00", "r00-01", "r01-00"]),
        held_out={UNSEEN: ids(["r02-00", "r02-01"]), SEEN: ids(["r01-01"])},
    )
    sets = {TRAIN: ids(f["id"][:6]), "test": ids(f["id"][6:])}
    return Splits(sets, (fold,)), f


def test_clean_splits_pass() -> None:
    splits, f = _clean()

    assert leak_errors(splits, f, id_column="id", group="group", never_held_out={"g00"}) == []


@pytest.mark.parametrize(
    ("sets", "fold", "protected", "message"),
    [
        ({"test": ["r00-00", "r03-00"]}, None, None, "both train and test"),
        (None, {"train": ["r00-00", "r02-00"]}, None, "both trained on and held out"),
        (None, {"train": ["r00-00", "r02-01"]}, None, "unseen groups in training rows"),
        (None, {"train": ["r00-00", "r03-01"]}, None, "not in train"),
        (None, None, {"g02"}, "protected groups held out"),
    ],
)
def test_leak_checks_catch_constructed_leaks(
    sets: dict[str, list[str]] | None,
    fold: dict[str, list[str]] | None,
    protected: set[str] | None,
    message: str,
) -> None:
    splits, f = _clean()
    new_sets = {**splits.sets, **{k: ids(v) for k, v in (sets or {}).items()}}
    new_fold = splits.folds[0]
    if fold:
        new_fold = Fold(train=ids(fold["train"]), held_out=new_fold.held_out)
    if fold and "r02-00" in fold["train"]:  # also hold it out: trained on and held out
        new_fold = Fold(train=new_fold.train, held_out={UNSEEN: ids(["r02-00"])})
    leaky = Splits(new_sets, (new_fold,))

    errors = leak_errors(leaky, f, id_column="id", group="group", never_held_out=protected)

    assert any(message in e for e in errors), errors

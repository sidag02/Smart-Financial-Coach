"""The Technical Design's baseline for spending spikes: per-user mean ± k·std on monthly spend.

Each period's spend is compared with the mean and standard deviation of the same user's spend in
that category over all earlier months (`history_mean`, `history_sd`). The score is the z; k is
the cutoff `SpikeThresholded` places. It is the PRD's "simple rule-based alternative" that a
candidate must beat at the same flag rate (FR-8 §5, decision 5). Periods without a spread score
-inf: there's nothing to compare with.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.models.registry import register
from smart_financial_coach.intelligence.spikes.threshold import SpikeScorerModel


@register("spending_spikes/mean_k_std")
class MeanKStd(SpikeScorerModel):
    def __init__(self) -> None:
        super().__init__()

    def scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        sd = x["history_sd"].to_numpy(dtype=float)
        z = (x["spend"].to_numpy(dtype=float) - x["history_mean"].to_numpy(dtype=float)) / np.where(
            sd > 0, sd, np.nan
        )
        return np.where(np.isfinite(z), z, -np.inf)

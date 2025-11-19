import numpy as np
import pandas as pd
from dataclasses import dataclass

def _ensure_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "week" not in df.columns:
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@dataclass
class WeekwiseMap:
    a_global: float
    b_global: float
    a_week: np.ndarray   # length 54, use 1..53
    b_week: np.ndarray
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp
    use_log1p: bool

def _smooth_week_params(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Smooth entries 1..53 with a simple moving average; arr[0] left as-is.
    """
    out = arr.copy()
    vals = arr[1:54]          # weeks 1..53
    if len(vals) == 0:
        return out
    k = kernel_size
    # pad at edges with nearest values
    padded = np.r_[vals[0], vals, vals[-1]]
    sm = np.convolve(padded, np.ones(k)/k, mode="same")[1:-1]
    out[1:54] = sm
    return out

def fit_weekwise_mapping(
    df_long: pd.DataFrame,     # long history (red)
    df_short: pd.DataFrame,    # short history (blue)
    state_cols,
    shrink_strength: float = 5.0,
    use_log1p: bool = False,
    smooth_kernel_size: int = 3,   # <= NEW: smooth a_week, b_week over weeks
) -> dict[str, WeekwiseMap]:
    """Learn per-week-of-year linear maps y_blue ≈ a_w * x_red + b_w with EB shrinkage."""
    state_cols = list(state_cols)
    dfL = _ensure_week(df_long)
    dfS = _ensure_week(df_short)

    # align on date+week for overlap
    J = pd.merge(
        dfL[["date", "week"] + state_cols],
        dfS[["date", "week"] + state_cols],
        on=["date", "week"], how="inner", suffixes=("_L", "_S")
    )
    if J.empty:
        raise ValueError("No overlapping dates between long and short series.")

    # pick last contiguous block with data in both
    mask_any = np.zeros(len(J), dtype=bool)
    for s in state_cols:
        mask_any |= (~J[f"{s}_L"].isna().values) & (~J[f"{s}_S"].isna().values)
    m = mask_any.astype(int)
    diff = np.diff(np.r_[0, m, 0])
    starts = np.where(diff == 1)[0]
    stops  = np.where(diff == -1)[0]
    if len(starts) == 0:
        raise ValueError("No contiguous overlap with data in both series.")
    s0, e0 = starts[-1], stops[-1]
    J = J.iloc[s0:e0].reset_index(drop=True)

    overlap_start = pd.Timestamp(J["date"].iloc[0])
    overlap_end   = pd.Timestamp(J["date"].iloc[-1])

    def fwd(x):
        if use_log1p:
            return np.log1p(np.clip(x, a_min=0, a_max=None).astype(float))
        return x.astype(float)

    maps: dict[str, WeekwiseMap] = {}

    for s in state_cols:
        x = fwd(J[f"{s}_L"].to_numpy())
        y = fwd(J[f"{s}_S"].to_numpy())
        w = J["week"].to_numpy().astype(int)

        keep = ~(np.isnan(x) | np.isnan(y))
        x, y, w = x[keep], y[keep], w[keep]
        if len(y) < 8:
            a0, b0 = 1.0, 0.0
            a_week = np.full(54, a0, dtype=float)
            b_week = np.full(54, b0, dtype=float)
            maps[s] = WeekwiseMap(a0, b0, a_week, b_week, overlap_start, overlap_end, use_log1p)
            continue

        # global OLS
        Xg = np.c_[x, np.ones_like(x)]
        try:
            beta_g, *_ = np.linalg.lstsq(Xg, y, rcond=None)
            a0, b0 = float(beta_g[0]), float(beta_g[1])
        except Exception:
            a0, b0 = 1.0, 0.0

        a_week = np.full(54, a0, dtype=float)
        b_week = np.full(54, b0, dtype=float)

        for wk in np.unique(w):
            idx = np.where(w == wk)[0]
            if idx.size < 2:
                continue
            xw, yw = x[idx], y[idx]
            Xw = np.c_[xw, np.ones_like(xw)]
            try:
                beta_w, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                aw, bw = float(beta_w[0]), float(beta_w[1])
            except Exception:
                aw, bw = a0, b0
            n = float(idx.size)
            lam = float(shrink_strength)
            a_week[wk] = (n * aw + lam * a0) / (n + lam)
            b_week[wk] = (n * bw + lam * b0) / (n + lam)

        # smooth across adjacent weeks to avoid week-to-week kinks
        if smooth_kernel_size and smooth_kernel_size > 1:
            a_week = _smooth_week_params(a_week, smooth_kernel_size)
            b_week = _smooth_week_params(b_week, smooth_kernel_size)

        maps[s] = WeekwiseMap(a0, b0, a_week, b_week, overlap_start, overlap_end, use_log1p)

    return maps

# def fit_weekwise_mapping(
#     df_long: pd.DataFrame,     # long history (red)
#     df_short: pd.DataFrame,    # short history (blue)
#     state_cols,                # list-like; will be coerced to list
#     shrink_strength: float = 5.0,
#     use_log1p: bool = False,
# ) -> dict[str, WeekwiseMap]:
#     """Learn per-week-of-year linear maps y_blue ≈ a_w * x_red + b_w with EB shrinkage."""
#     # --- ensure types/order ---
#     state_cols = list(state_cols)  # <-- FIX: avoid Index arithmetic
#     dfL = _ensure_week(df_long)
#     dfS = _ensure_week(df_short)

#     # align on date+week for overlap
#     J = pd.merge(
#         dfL[["date", "week"] + state_cols],
#         dfS[["date", "week"] + state_cols],
#         on=["date", "week"], how="inner", suffixes=("_L", "_S")
#     )
#     if J.empty:
#         raise ValueError("No overlapping dates between long and short series.")

#     # pick last contiguous block with any data present in both
#     mask_any = np.zeros(len(J), dtype=bool)
#     for s in state_cols:
#         mask_any |= (~J[f"{s}_L"].isna().values) & (~J[f"{s}_S"].isna().values)
#     m = mask_any.astype(int)
#     diff = np.diff(np.r_[0, m, 0])
#     starts = np.where(diff == 1)[0]
#     stops  = np.where(diff == -1)[0]
#     if len(starts) == 0:
#         raise ValueError("No contiguous overlap with data in both series.")
#     s0, e0 = starts[-1], stops[-1]
#     J = J.iloc[s0:e0].reset_index(drop=True)

#     overlap_start = pd.Timestamp(J["date"].iloc[0])
#     overlap_end   = pd.Timestamp(J["date"].iloc[-1])

#     def fwd(x):
#         if use_log1p:
#             return np.log1p(np.clip(x, a_min=0, a_max=None).astype(float))
#         return x.astype(float)

#     maps: dict[str, WeekwiseMap] = {}

#     for s in state_cols:
#         x = fwd(J[f"{s}_L"].to_numpy())
#         y = fwd(J[f"{s}_S"].to_numpy())
#         w = J["week"].to_numpy().astype(int)

#         keep = ~(np.isnan(x) | np.isnan(y))
#         x, y, w = x[keep], y[keep], w[keep]
#         if len(y) < 8:
#             a0, b0 = 1.0, 0.0
#             a_week = np.full(54, a0, dtype=float)
#             b_week = np.full(54, b0, dtype=float)
#             maps[s] = WeekwiseMap(a0, b0, a_week, b_week, overlap_start, overlap_end, use_log1p)
#             continue

#         # global OLS
#         Xg = np.c_[x, np.ones_like(x)]
#         try:
#             beta_g, *_ = np.linalg.lstsq(Xg, y, rcond=None)
#             a0, b0 = float(beta_g[0]), float(beta_g[1])
#         except Exception:
#             a0, b0 = 1.0, 0.0

#         a_week = np.full(54, a0, dtype=float)
#         b_week = np.full(54, b0, dtype=float)

#         for wk in np.unique(w):
#             idx = np.where(w == wk)[0]
#             if idx.size < 2:
#                 continue
#             xw, yw = x[idx], y[idx]
#             Xw = np.c_[xw, np.ones_like(xw)]
#             try:
#                 beta_w, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
#                 aw, bw = float(beta_w[0]), float(beta_w[1])
#             except Exception:
#                 aw, bw = a0, b0
#             n = float(idx.size)
#             lam = float(shrink_strength)
#             a_week[wk] = (n * aw + lam * a0) / (n + lam)
#             b_week[wk] = (n * bw + lam * b0) / (n + lam)

#         maps[s] = WeekwiseMap(a0, b0, a_week, b_week, overlap_start, overlap_end, use_log1p)

#     return maps

def despike_isolated_points(
    series: pd.Series,
    max_rel_drop: float = 0.5,
    max_rel_jump: float = 0.5,
) -> pd.Series:
    """
    Fix isolated one-week dips/spikes:
    - If y_t is much lower than both neighbors -> pull it up toward their mean.
    - If y_t is much higher than both neighbors -> pull it down toward their mean.
    max_rel_drop/jump are thresholds on relative change (0.5 = 50%).
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 3:
        return series

    out = y.copy()
    for t in range(1, n - 1):
        y_prev, y_curr, y_next = y[t-1], y[t], y[t+1]
        # ignore NaNs
        if np.isnan(y_prev) or np.isnan(y_curr) or np.isnan(y_next):
            continue
        local_mean = 0.5 * (y_prev + y_next) if (y_prev + y_next) > 0 else y_curr

        # isolated dip
        if y_curr < y_prev and y_curr < y_next:
            if local_mean > 0 and y_curr < (1 - max_rel_drop) * local_mean:
                out[t] = local_mean

        # isolated spike
        if y_curr > y_prev and y_curr > y_next:
            if local_mean > 0 and y_curr > (1 + max_rel_jump) * local_mean:
                out[t] = local_mean

    return pd.Series(out, index=series.index, name=series.name)


def apply_weekwise_mapping_to_preoverlap(
    df_long: pd.DataFrame,
    maps: dict[str, WeekwiseMap],
    state_cols,
    despike: bool = True,
) -> pd.DataFrame:
    """
    Apply learned week-wise map to dates < overlap_start; keep others unchanged.
    Also optionally despikes only the pre-overlap segment.

    This version avoids SettingWithCopyWarning by never assigning to a slice
    of a Series — only to df.loc[...] directly.
    """
    state_cols = list(state_cols)
    df = _ensure_week(df_long).copy()

    for s in state_cols:
        if s not in df.columns or s not in maps:
            continue
        m = maps[s]
        pre_mask = df["date"] < m.overlap_start
        if not pre_mask.any():
            continue

        vals = df[s].astype(float).to_numpy()
        weeks = df["week"].to_numpy().astype(int)

        # forward transform space used in fitting
        if m.use_log1p:
            base = np.log1p(np.clip(vals, a_min=0, a_max=None))
        else:
            base = vals.copy()

        out = vals.copy()
        pre_idx = np.where(pre_mask.values)[0]
        for i in pre_idx:
            wk = weeks[i]
            if wk < 1 or wk > 53:
                wk = 53
            a = m.a_week[wk]
            b = m.b_week[wk]
            z = a * base[i] + b
            out[i] = np.expm1(z) if m.use_log1p else z

        # non-negative
        out = np.clip(out, a_min=0.0, a_max=None)

        # write full column once → no chained assignment
        df[s] = out

        # optional: despike only pre-overlap segment
        if despike:
            pre_vals = df.loc[pre_mask, s]
            despiked = despike_isolated_points(pre_vals)
            # assign back via df.loc, not via a Series slice
            df.loc[pre_mask, s] = despiked.to_numpy()

    return df



#usage
# maps = fit_weekwise_mapping(df_ili0, df_hosp, list(states), shrink_strength=5.0, use_log1p=False)
# df_ili = apply_weekwise_mapping_to_preoverlap(df_ili0, maps, list(states))
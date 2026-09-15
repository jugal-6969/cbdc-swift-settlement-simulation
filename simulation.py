"""
MSO4992 - Chapter 5 Simulation
Evaluating CBDC-based settlement predictability vs the SWIFT correspondent rail.

This script implements the model specified in Chapter 3 (Methodology):
  - A SWIFT-style correspondent rail: each payment is routed through a chain of
    intermediary banks, each contributing a delay; a beneficiary-side delay is
    then added that is larger and more variable for lower-income destinations.
    Consistent with Nilsson et al. (2022), additional intermediaries raise both
    the level and the VARIANCE of elapsed time (here via per-hop "stall" events
    representing offline hours / compliance holds).
  - A CBDC rail: direct settlement on a shared ledger, near-real-time, with low
    variance and no intermediary chain (calibrated to Project mBridge design).

Two controlled variables: number of intermediaries and destination income band.
Calibrated so the SWIFT rail reproduces the published SWIFT gpi benchmark
(mean ~8h36m = 516 min; median ~1h38m = 98 min; Nilsson et al., 2022).

All randomness is governed by a single fixed seed for exact reproducibility.

Author: Jugal Bhagat
"""

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

# --------------------------------------------------------------------------
# 0. Global configuration
# --------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_PER_CELL = 20_000          # simulated payments per (intermediaries x income) cell, per rail
INTERMEDIARY_LEVELS = [1, 2, 3, 4, 5]
INCOME_BANDS = ["High", "Upper-middle", "Lower-middle", "Low"]

# SWIFT gpi calibration targets (Nilsson et al., 2022), in minutes
TARGET_MEAN_MIN = 8 * 60 + 36     # 516
TARGET_MEDIAN_MIN = 1 * 60 + 38   # 98

# --------------------------------------------------------------------------
# 1. Model parameters (in minutes)
#    These are the calibration constants. They are tuned so that a realistic
#    global mix of payments reproduces the gpi mean/median above. Every value
#    is an explicit, documented assumption.
# --------------------------------------------------------------------------

# Per-intermediary "base" processing delay (small), lognormal
HOP_BASE_MEANLOG = np.log(8.0)    # median ~8 min per hop
HOP_BASE_SDLOG   = 0.8

# Per-intermediary "stall" event: offline hours / compliance hold (rare, large).
# Probability applies PER HOP, so more hops -> more chance of a stall -> fatter
# right tail -> higher variance. This is the mechanism behind the intermediary
# effect on variance (Nilsson et al., 2022: capital controls, offline hours).
HOP_STALL_PROB    = 0.17
HOP_STALL_MEANLOG = np.log(760.0)  # median ~12.7 h when a stall occurs (offline/hold)
HOP_STALL_SDLOG   = 0.7

# Beneficiary-leg delay by destination income band (dominant component).
# Lower income -> larger location AND larger spread (income is the single
# strongest predictor of processing time; Nilsson et al., 2022).
BENEFICIARY = {
    "High":         {"meanlog": np.log(12.0),  "sdlog": 0.9},
    "Upper-middle": {"meanlog": np.log(35.0),  "sdlog": 1.15},
    "Lower-middle": {"meanlog": np.log(80.0),  "sdlog": 1.35},
    "Low":          {"meanlog": np.log(260.0), "sdlog": 1.60},
}

# CBDC rail: direct settlement, near-real-time. Modelled as a small delay with
# low variance, invariant to intermediaries and income (there is no
# intermediary chain and no correspondent beneficiary processing).
CBDC_MEANLOG = np.log(0.5)   # median ~30 seconds
CBDC_SDLOG   = 0.35          # low variance

# Real-world global mix used ONLY to validate the gpi calibration: weights for
# how common each (intermediaries, income) combination is worldwide. Fast
# high-income low-hop payments are common; slow low-income high-hop ones are
# rarer but create the fat tail. (Used for the calibration check only.)
INCOME_MIX = {"High": 0.42, "Upper-middle": 0.30, "Lower-middle": 0.22, "Low": 0.06}
INTERM_MIX = {1: 0.30, 2: 0.34, 3: 0.22, 4: 0.10, 5: 0.04}

# The four study corridors, mapped to a representative income band and a
# representative intermediary count. NOTE: the World Bank data does not record
# intermediary count, so these counts are informed assumptions the simulation
# exists to reason about -- adjust if your methodology intends different values.
CORRIDORS = {
    "US->India":     {"income": "Lower-middle", "intermediaries": 2},
    "US->China":     {"income": "Upper-middle", "intermediaries": 2},
    "UAE->India":    {"income": "Lower-middle", "intermediaries": 3},
    "UAE->Pakistan": {"income": "Lower-middle", "intermediaries": 4},
}


# --------------------------------------------------------------------------
# 2. Core payment-time samplers
# --------------------------------------------------------------------------
def swift_times(n, intermediaries, income_band, rng):
    """Vectorised SWIFT-rail end-to-end settlement times (minutes)."""
    total = np.zeros(n)
    # intermediary chain
    for _ in range(intermediaries):
        base = rng.lognormal(HOP_BASE_MEANLOG, HOP_BASE_SDLOG, n)
        stalled = rng.random(n) < HOP_STALL_PROB
        stall = np.where(
            stalled,
            rng.lognormal(HOP_STALL_MEANLOG, HOP_STALL_SDLOG, n),
            0.0,
        )
        total += base + stall
    # beneficiary leg (income-dependent, dominant)
    b = BENEFICIARY[income_band]
    total += rng.lognormal(b["meanlog"], b["sdlog"], n)
    return total


def cbdc_times(n, rng):
    """Vectorised CBDC-rail settlement times (minutes) -- direct, near real-time."""
    return rng.lognormal(CBDC_MEANLOG, CBDC_SDLOG, n)


# --------------------------------------------------------------------------
# 3. Calibration check against the SWIFT gpi benchmark
# --------------------------------------------------------------------------
def calibration_check(rng):
    """Simulate a realistic global mix and compare to the gpi mean/median."""
    parts = []
    n_cal = 400_000
    for inc, w_i in INCOME_MIX.items():
        for k, w_k in INTERM_MIX.items():
            m = int(round(n_cal * w_i * w_k))
            if m > 0:
                parts.append(swift_times(m, k, inc, rng))
    pooled = np.concatenate(parts)
    return {
        "n": int(pooled.size),
        "mean_min": float(np.mean(pooled)),
        "median_min": float(np.median(pooled)),
        "target_mean_min": TARGET_MEAN_MIN,
        "target_median_min": TARGET_MEDIAN_MIN,
    }


# --------------------------------------------------------------------------
# 4. Build the full simulated dataset (grid of intermediaries x income)
# --------------------------------------------------------------------------
def build_dataset(rng):
    rows = []
    for k in INTERMEDIARY_LEVELS:
        for inc in INCOME_BANDS:
            s = swift_times(N_PER_CELL, k, inc, rng)
            c = cbdc_times(N_PER_CELL, rng)
            for t in s:
                rows.append((k, inc, "SWIFT", t))
            for t in c:
                rows.append((k, inc, "CBDC", t))
    return pd.DataFrame(rows, columns=["intermediaries", "income", "rail", "time_min"])


# --------------------------------------------------------------------------
# 5. Effect-size helpers
# --------------------------------------------------------------------------
def epsilon_squared_kw(H, n, k_groups):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return float((H - k_groups + 1) / (n - k_groups))


def cliffs_delta(a, b):
    """Cliff's delta via the Mann-Whitney U (memory-safe)."""
    a = np.asarray(a); b = np.asarray(b)
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * U / (len(a) * len(b)) - 1.0)


# --------------------------------------------------------------------------
# 6. Run everything
# --------------------------------------------------------------------------
def main():
    results = {}

    # --- calibration ---
    results["calibration"] = calibration_check(rng)

    # --- dataset ---
    df = build_dataset(rng)
    df.to_csv("simulated_settlement_times.csv", index=False)

    swift_all = df[df.rail == "SWIFT"].time_min.values
    cbdc_all = df[df.rail == "CBDC"].time_min.values

    # --- RQ1: headline SWIFT vs CBDC (pooled) ---
    H, p = stats.kruskal(swift_all, cbdc_all)
    lev_stat, lev_p = stats.levene(swift_all, cbdc_all, center="median")
    results["rq1"] = {
        "swift": {"mean": float(np.mean(swift_all)), "median": float(np.median(swift_all)),
                   "sd": float(np.std(swift_all, ddof=1)),
                   "iqr": float(np.subtract(*np.percentile(swift_all, [75, 25])))},
        "cbdc": {"mean": float(np.mean(cbdc_all)), "median": float(np.median(cbdc_all)),
                  "sd": float(np.std(cbdc_all, ddof=1)),
                  "iqr": float(np.subtract(*np.percentile(cbdc_all, [75, 25])))},
        "kruskal_H": float(H), "kruskal_p": float(p),
        "kruskal_eps2": epsilon_squared_kw(H, len(swift_all) + len(cbdc_all), 2),
        "levene_stat": float(lev_stat), "levene_p": float(lev_p),
        "variance_ratio": float(np.var(swift_all, ddof=1) / np.var(cbdc_all, ddof=1)),
        "cliffs_delta": cliffs_delta(swift_all, cbdc_all),
        "sd_ratio": float(np.std(swift_all, ddof=1) / np.std(cbdc_all, ddof=1)),
    }

    # --- Effect of intermediary count (SWIFT only) ---
    inter = {}
    swift = df[df.rail == "SWIFT"]
    groups = [swift[swift.intermediaries == k].time_min.values for k in INTERMEDIARY_LEVELS]
    H_i, p_i = stats.kruskal(*groups)
    lev_i, levp_i = stats.levene(*groups, center="median")
    for k in INTERMEDIARY_LEVELS:
        g = swift[swift.intermediaries == k].time_min.values
        inter[k] = {"mean": float(np.mean(g)), "median": float(np.median(g)),
                    "sd": float(np.std(g, ddof=1)),
                    "iqr": float(np.subtract(*np.percentile(g, [75, 25])))}
    results["intermediary_effect"] = {
        "by_level": inter, "kruskal_H": float(H_i), "kruskal_p": float(p_i),
        "kruskal_eps2": epsilon_squared_kw(H_i, len(swift), len(INTERMEDIARY_LEVELS)),
        "levene_stat": float(lev_i), "levene_p": float(levp_i),
    }

    # --- Effect of income band (SWIFT only) ---
    income = {}
    groups_inc = [swift[swift.income == b].time_min.values for b in INCOME_BANDS]
    H_b, p_b = stats.kruskal(*groups_inc)
    lev_b, levp_b = stats.levene(*groups_inc, center="median")
    for b in INCOME_BANDS:
        g = swift[swift.income == b].time_min.values
        income[b] = {"mean": float(np.mean(g)), "median": float(np.median(g)),
                     "sd": float(np.std(g, ddof=1)),
                     "iqr": float(np.subtract(*np.percentile(g, [75, 25])))}
    # Dunn's post-hoc (Holm) across income bands, as promised in Chapter 3
    dunn = sp.posthoc_dunn(
        [swift[swift.income == b].time_min.values for b in INCOME_BANDS],
        p_adjust="holm",
    )
    dunn.index = INCOME_BANDS; dunn.columns = INCOME_BANDS
    results["income_effect"] = {
        "by_band": income, "kruskal_H": float(H_b), "kruskal_p": float(p_b),
        "kruskal_eps2": epsilon_squared_kw(H_b, len(swift), len(INCOME_BANDS)),
        "levene_stat": float(lev_b), "levene_p": float(levp_b),
        "dunn_holm": dunn.round(4).to_dict(),
    }

    # --- RQ2: THRESHOLD ANALYSIS (the core contribution) ---
    # For each (intermediaries x income) cell: the predictability gap between
    # rails, measured as the RATIO of SWIFT SD to CBDC SD (an effect size), plus
    # the absolute SWIFT SD in minutes (the operational unpredictability), and
    # Levene's p (reported but not the headline -- see note).
    grid_sd_ratio = pd.DataFrame(index=INCOME_BANDS, columns=INTERMEDIARY_LEVELS, dtype=float)
    grid_swift_sd = pd.DataFrame(index=INCOME_BANDS, columns=INTERMEDIARY_LEVELS, dtype=float)
    grid_swift_iqr = pd.DataFrame(index=INCOME_BANDS, columns=INTERMEDIARY_LEVELS, dtype=float)
    grid_levene_p = pd.DataFrame(index=INCOME_BANDS, columns=INTERMEDIARY_LEVELS, dtype=float)
    for k in INTERMEDIARY_LEVELS:
        for b in INCOME_BANDS:
            s = df[(df.rail == "SWIFT") & (df.intermediaries == k) & (df.income == b)].time_min.values
            c = df[(df.rail == "CBDC") & (df.intermediaries == k) & (df.income == b)].time_min.values
            s_sd = np.std(s, ddof=1); c_sd = np.std(c, ddof=1)
            grid_sd_ratio.loc[b, k] = s_sd / c_sd
            grid_swift_sd.loc[b, k] = s_sd
            grid_swift_iqr.loc[b, k] = np.subtract(*np.percentile(s, [75, 25]))
            _, lp = stats.levene(s, c, center="median")
            grid_levene_p.loc[b, k] = lp
    results["threshold"] = {
        "sd_ratio": grid_sd_ratio.round(2).to_dict(),
        "swift_sd_min": grid_swift_sd.round(1).to_dict(),
        "swift_iqr_min": grid_swift_iqr.round(1).to_dict(),
        "levene_p": {k: {b: f"{grid_levene_p.loc[b, k]:.2e}" for b in INCOME_BANDS}
                      for k in INTERMEDIARY_LEVELS},
    }

    # --- Corridor-level summary ---
    corridor_rows = {}
    for name, cfg in CORRIDORS.items():
        s = df[(df.rail == "SWIFT") & (df.intermediaries == cfg["intermediaries"]) &
               (df.income == cfg["income"])].time_min.values
        c = cbdc_times(N_PER_CELL, rng)
        corridor_rows[name] = {
            "income": cfg["income"], "intermediaries": cfg["intermediaries"],
            "swift_median": float(np.median(s)), "swift_sd": float(np.std(s, ddof=1)),
            "swift_iqr": float(np.subtract(*np.percentile(s, [75, 25]))),
            "cbdc_median": float(np.median(c)), "cbdc_sd": float(np.std(c, ddof=1)),
            "sd_ratio": float(np.std(s, ddof=1) / np.std(c, ddof=1)),
        }
    results["corridors"] = corridor_rows

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    make_figures(df, results, grid_sd_ratio, grid_swift_sd, grid_swift_iqr, inter, income)
    print(json.dumps(results, indent=2))
    return results


# --------------------------------------------------------------------------
# 7. Figures
# --------------------------------------------------------------------------
def make_figures(df, results, grid_sd_ratio, grid_swift_sd, grid_swift_iqr, inter, income):
    mpl.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True,
                          "grid.alpha": 0.3, "axes.spines.top": False,
                          "axes.spines.right": False})
    C_SWIFT = "#c0392b"; C_CBDC = "#2471a3"

    # Fig 5.1 -- distribution comparison, stacked panels sharing a log x-axis
    s = df[df.rail == "SWIFT"].time_min.values
    c = df[df.rail == "CBDC"].time_min.values
    bins = np.logspace(-1.3, np.log10(max(s.max(), 10)), 70)
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axt.hist(s, bins=bins, color=C_SWIFT, alpha=0.85)
    axt.set_xscale("log"); axt.set_ylabel("Count"); axt.set_title("SWIFT rail")
    axt.axvline(np.median(s), color="black", ls="--", lw=1)
    axt.text(np.median(s) * 1.1, axt.get_ylim()[1] * 0.8,
             f"median {np.median(s):.0f} min", fontsize=9)
    axb.hist(c, bins=bins, color=C_CBDC, alpha=0.85)
    axb.set_xscale("log"); axb.set_ylabel("Count"); axb.set_title("CBDC rail")
    axb.axvline(np.median(c), color="black", ls="--", lw=1)
    axb.text(np.median(c) * 1.3, axb.get_ylim()[1] * 0.8,
             f"median {np.median(c)*60:.0f} sec", fontsize=9)
    axb.set_xlabel("Settlement time (minutes, log scale)")
    fig.suptitle("Distribution of settlement time by rail (all conditions pooled)")
    fig.tight_layout(); fig.savefig("fig_5_1_distributions.png", dpi=150); plt.close(fig)

    # Fig 5.2 -- intermediary effect on SWIFT median and variability (IQR)
    ks = INTERMEDIARY_LEVELS
    med = [inter[k]["median"] for k in ks]
    iqr = [inter[k]["iqr"] for k in ks]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(ks, med, "o-", color=C_SWIFT, label="Median settlement time")
    ax1.set_xlabel("Number of intermediaries"); ax1.set_ylabel("Median (minutes)", color=C_SWIFT)
    ax1.tick_params(axis="y", labelcolor=C_SWIFT); ax1.set_xticks(ks)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(ks, iqr, "s--", color="#6c3483", label="IQR (spread)")
    ax2.set_ylabel("IQR (minutes)", color="#6c3483"); ax2.tick_params(axis="y", labelcolor="#6c3483")
    ax1.set_title("SWIFT rail: settlement time and spread rise with intermediary count")
    fig.tight_layout(); fig.savefig("fig_5_2_intermediaries.png", dpi=150); plt.close(fig)

    # Fig 5.3 -- income effect on SWIFT (median with IQR bars)
    bands = INCOME_BANDS
    med_b = [income[b]["median"] for b in bands]
    iqr_b = [income[b]["iqr"] for b in bands]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(bands))
    ax.bar(x, med_b, color=C_SWIFT, alpha=0.75, label="Median")
    ax.errorbar(x, med_b, yerr=[np.zeros(len(bands)), iqr_b], fmt="none",
                ecolor="#7b241c", capsize=5, label="IQR (upward)")
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("Settlement time (minutes)")
    ax.set_title("SWIFT rail: cost of settlement rises as destination income falls")
    ax.legend()
    fig.tight_layout(); fig.savefig("fig_5_3_income.png", dpi=150); plt.close(fig)

    # Fig 5.4 -- THRESHOLD heatmap: SWIFT IQR (robust unpredictability) across grid
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = grid_swift_iqr.astype(float).values
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(INTERMEDIARY_LEVELS))); ax.set_xticklabels(INTERMEDIARY_LEVELS)
    ax.set_yticks(range(len(INCOME_BANDS))); ax.set_yticklabels(INCOME_BANDS)
    ax.set_xlabel("Number of intermediaries"); ax.set_ylabel("Destination income band")
    ax.set_title("Predictability gap: SWIFT settlement-time IQR (minutes) by condition")
    for i in range(len(INCOME_BANDS)):
        for j in range(len(INTERMEDIARY_LEVELS)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                    color="black" if data[i, j] < data.max() * 0.6 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="SWIFT IQR (minutes)")
    fig.tight_layout(); fig.savefig("fig_5_4_threshold_heatmap.png", dpi=150); plt.close(fig)

    print("Figures written: fig_5_1..fig_5_4")


if __name__ == "__main__":
    main()

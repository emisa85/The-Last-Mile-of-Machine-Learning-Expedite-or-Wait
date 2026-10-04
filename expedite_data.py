"""
Synthetic inbound-shipment data for the "Expedite or Wait?" lesson.

Happy Valley Gear Co. (a made-up company) runs a distribution center in
Bellefonte, PA, just off I-99 / I-80, and supplies game-day gear to stores
around State College.  Every inbound shipment gets a probability of arriving
late from two models:

  Vendor model    - gradient boosting on 40 features  (the "fancy" model)
  Homemade model  - logistic regression on 5 features plus the carrier's
                    "behind schedule" alert flag, built by the dock team

Everything is generated from a fixed seed so the deck, the lecture notes and
the Streamlit tool all show the same numbers.
"""
import numpy as np
import pandas as pd

SEED = 814  # the State College area code

ORIGINS = [  # city, approx road miles to Bellefonte PA, share of shipments
    ("Harrisburg, PA", 90, 0.16),
    ("Pittsburgh, PA", 140, 0.14),
    ("Philadelphia, PA", 200, 0.14),
    ("Newark, NJ", 260, 0.10),
    ("Columbus, OH", 300, 0.10),
    ("Charlotte, NC", 480, 0.08),
    ("Chicago, IL", 600, 0.10),
    ("Atlanta, GA", 720, 0.07),
    ("Dallas, TX", 1400, 0.06),
    ("Los Angeles, CA", 2600, 0.05),
]
CARRIERS = [  # made-up carrier, effect on the log-odds of being late
    ("Keystone Freight", -0.45),
    ("Allegheny Express", 0.00),
    ("Susquehanna Trucking", 0.55),
    ("Blue Ridge Lines", 1.05),
]
MODES = [("Full truckload", -0.25, 0.45), ("LTL", 0.50, 0.40), ("Parcel", 0.00, 0.15)]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PER_WEEK = 400  # inbound shipments per week at the Bellefonte DC


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt(y, score):
    """Two-parameter (Platt) calibration fitted by Newton steps. Returns calibrated probs."""
    x = _logit(score)
    a, b = 1.0, 0.0
    for _ in range(50):
        z = a * x + b
        p = _sigmoid(z)
        w = p * (1 - p)
        g_a = np.sum((p - y) * x)
        g_b = np.sum(p - y)
        h_aa = np.sum(w * x * x) + 1e-9
        h_ab = np.sum(w * x)
        h_bb = np.sum(w) + 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
    return _sigmoid(a * x + b)


def make_data(n=2000, seed=SEED):
    rng = np.random.default_rng(seed)

    o_idx = rng.choice(len(ORIGINS), size=n, p=[w for _, _, w in ORIGINS])
    origin = np.array([ORIGINS[i][0] for i in o_idx])
    miles = np.array([ORIGINS[i][1] for i in o_idx], dtype=float)
    miles = np.round(miles * rng.uniform(0.92, 1.08, n)).astype(int)

    c_idx = rng.choice(len(CARRIERS), size=n, p=[0.35, 0.30, 0.20, 0.15])
    carrier = np.array([CARRIERS[i][0] for i in c_idx])
    carrier_eff = np.array([CARRIERS[i][1] for i in c_idx])

    m_idx = rng.choice(len(MODES), size=n, p=[m[2] for m in MODES])
    mode = np.array([MODES[i][0] for i in m_idx])
    mode_eff = np.array([MODES[i][1] for i in m_idx])

    week = rng.integers(1, 6, size=n)  # five weeks of history
    weekday = rng.choice(WEEKDAYS, size=n, p=[0.24, 0.20, 0.19, 0.19, 0.18])
    snow = rng.random(n) < 0.18  # winter in Happy Valley
    lead_time = np.clip(np.round(miles / 450 + rng.normal(1.6, 0.8, n)), 1, 9).astype(int)
    supplier_otp = np.clip(np.round(rng.normal(88, 7, n)), 60, 99).astype(int)
    tight = (lead_time <= np.maximum(1, np.round(miles / 500))).astype(float)

    # The truth: features the vendor model sees well, plus things nobody measures
    logit_feat = (
        -4.70
        + 0.00105 * miles
        + carrier_eff
        + mode_eff
        + 1.55 * snow
        + 0.35 * (weekday == "Mon")
        - 0.055 * (supplier_otp - 88)
        + 0.65 * tight
    )
    hidden = rng.normal(0, 1.3, n)  # driver, dock congestion, paperwork ... unmeasured
    p_true = _sigmoid(logit_feat + hidden)
    late = rng.random(n) < p_true

    # ---- Vendor model: sees the features well and part of the hidden stuff ----
    raw_vendor = _sigmoid(logit_feat + 0.92 * hidden + rng.normal(0, 0.15, n))
    p_vendor = _platt(late.astype(float), raw_vendor)

    # ---- Homemade model: 5 features (noisier) + the carrier's "behind schedule" alert ----
    alert = np.zeros(n, dtype=bool)
    alert[late] = rng.random(late.sum()) < 0.40          # the alert catches ~40% of lates
    alert[~late] = rng.random((~late).sum()) < 0.0045    # and rarely fires by mistake
    raw_home = _sigmoid(0.45 * logit_feat + 3.8 * alert + rng.normal(0, 1.3, n))
    p_home = _platt(late.astype(float), raw_home)

    df = pd.DataFrame(
        {
            "ship_id": [f"HV-{i + 1:04d}" for i in range(n)],
            "week": week,
            "arrival_day": weekday,
            "origin": origin,
            "miles": miles,
            "carrier": carrier,
            "mode": mode,
            "lead_time_days": lead_time,
            "snow_forecast": snow.astype(int),
            "supplier_on_time_pct": supplier_otp,
            "carrier_alert": alert.astype(int),
            "late": late.astype(int),
            "p_vendor": np.round(p_vendor, 3),
            "p_homemade": np.round(p_home, 3),
        }
    )
    return df


# ---------------------------------------------------------------- metrics ----
def confusion(y, score, t):
    """Counts for the rule 'expedite if score >= t'."""
    y = np.asarray(y)
    pred = np.asarray(score) >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics(y, score, t, fee, penalty, per_week=PER_WEEK):
    tp, fp, fn, tn = confusion(y, score, t)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else 0.0
    cost_total = fee * (tp + fp) + penalty * fn
    cost_week = cost_total / n * per_week
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=acc, precision=prec, recall=rec,
                expedited=tp + fp, cost_total=cost_total, cost_week=cost_week)


def metrics_topk(y, score, k, fee, penalty, per_week=PER_WEEK):
    """Expedite exactly the k highest-scoring shipments (a capacity cap)."""
    y = np.asarray(y)
    score = np.asarray(score)
    order = np.argsort(-score, kind="stable")
    pred = np.zeros(len(y), dtype=bool)
    pred[order[:k]] = True
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    n = len(y)
    cost_total = fee * (tp + fp) + penalty * fn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / k if k else float("nan"),
                recall=tp / (tp + fn), expedited=k, cost_total=cost_total,
                cost_week=cost_total / n * per_week, cutoff=float(score[order[k - 1]]) if k else 1.0)


def roc_curve(y, score):
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    ys = y[order]
    P, N = ys.sum(), len(ys) - ys.sum()
    tpr = np.concatenate([[0.0], np.cumsum(ys) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ys) / N])
    return fpr, tpr


def auc(y, score):
    fpr, tpr = roc_curve(y, score)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def cost_curve(y, score, fee, penalty, per_week=PER_WEEK, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    costs = np.array([metrics(y, score, t, fee, penalty, per_week)["cost_week"] for t in grid])
    return grid, costs


def best_threshold(y, score, fee, penalty, per_week=PER_WEEK):
    grid, costs = cost_curve(y, score, fee, penalty, per_week)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def formula_threshold(fee, penalty):
    """Expedite when p x penalty > fee, so the cutoff is p* = fee / penalty."""
    return min(1.0, fee / penalty) if penalty > 0 else 1.0


def calibration_table(y, score, edges=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    y = np.asarray(y)
    score = np.asarray(score)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score > lo) & (score <= hi) if lo > 0 else (score >= lo) & (score <= hi)
        if m.sum():
            rows.append(dict(bin=f"{lo:.2f}-{hi:.2f}", n=int(m.sum()),
                             avg_score=float(score[m].mean()), late_rate=float(y[m].mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = make_data()
    d.to_csv("shipments.csv", index=False)
    print(d.head())
    print("late rate", d.late.mean())

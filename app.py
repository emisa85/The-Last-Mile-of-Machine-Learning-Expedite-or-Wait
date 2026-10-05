"""
Expedite or Wait?  -  the classroom threshold tool
Run locally:   streamlit run app.py
Deploy free:   push app.py, expedite_data.py and requirements.txt to GitHub,
               then share.streamlit.io -> New app -> pick the repo -> Deploy.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from expedite_data import (make_data, metrics, metrics_topk, roc_curve, auc, cost_curve,
                           best_threshold, formula_threshold, calibration_table, PER_WEEK)

st.set_page_config(page_title="Expedite or Wait?", page_icon="🚚", layout="wide")

BLUE, ORANGE, SKY, NAVY, GRAY = "#2457A0", "#D26A00", "#009CDE", "#041E41", "#8A9199"
SCENARIOS = {
    "Game-day week  (fee \\$150, penalty \\$1,500)": (150, 1500),
    "Normal week  (fee \\$150, penalty \\$600)": (150, 600),
    "Air-freight expedite  (fee \\$600, penalty \\$900)": (600, 900),
    "Custom": None,
}
ROUND1 = ["HV-1766", "HV-0993", "HV-1770", "HV-0174", "HV-0253", "HV-0510", "HV-0324", "HV-1447", "HV-0175", "HV-1437"]

st.markdown(
    f"""
    <style>
      .big-number {{ font-size: 2.6rem; font-weight: 800; color: {NAVY}; line-height: 1.05; }}
      .small-label {{ font-size: 0.85rem; color: #4A5561; text-transform: uppercase; letter-spacing: .04em; }}
      .cm-cell {{ border-radius: 10px; padding: 14px 10px; text-align: center; color: white; }}
      .cm-title {{ font-size: 0.8rem; opacity: .9; text-transform: uppercase; letter-spacing: .04em; }}
      .cm-num {{ font-size: 2.0rem; font-weight: 800; line-height: 1.1; }}
      .cm-sub {{ font-size: 0.85rem; opacity: .95; }}
      .idea {{ background: #EAF4FB; border-radius: 10px; padding: 12px 16px; color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    return make_data()


d = load()
y = d.late.values

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🚚 Expedite or Wait?")
    st.caption("Happy Valley Gear Co. (a made-up company) - Bellefonte, PA distribution center")
    scenario = st.radio("Class scenario", list(SCENARIOS.keys()), index=0)
    if SCENARIOS[scenario]:
        fee, penalty = SCENARIOS[scenario]
        st.write(f"Expedite fee **\\${fee:,}**  ·  Late penalty **\\${penalty:,}**")
    else:
        fee = st.number_input("Expedite fee ($ per expedited shipment)", 0, 5000, 150, 25)
        penalty = st.number_input("Late penalty ($ per late shipment you did not expedite)", 0, 20000, 1500, 50)
    per_week = st.number_input("Inbound shipments per week", 50, 5000, PER_WEEK, 50)
    model_name = st.radio("Model", ["Vendor model (AUC 0.86)", "Homemade model (AUC 0.76)"], index=0)
    col = "p_vendor" if model_name.startswith("Vendor") else "p_homemade"
    score = d[col].values
    st.markdown("---")
    t = st.slider("Threshold  -  expedite if probability ≥ t", 0.01, 0.99, 0.50, 0.01)
    st.caption("Software default is 0.50. Is that where the money is?")
    st.markdown("---")
    st.caption("2,000 shipments = 5 weeks of history. Costs are shown per week.")

t_star = formula_threshold(fee, penalty)
m = metrics(y, score, t, fee, penalty, per_week)
m_never = metrics(y, score, 1.01, fee, penalty, per_week)
m_all = metrics(y, score, 0.0, fee, penalty, per_week)
m_star = metrics(y, score, t_star, fee, penalty, per_week)
grid, costs = cost_curve(y, score, fee, penalty, per_week)
bt, bc = best_threshold(y, score, fee, penalty, per_week)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Play with the threshold", "Round 1: ten shipments", "Model showdown", "Is the model honest?", "Data"]
)

# ------------------------------------------------------------- tab 1 -------
with tab1:
    st.subheader(f"{model_name.split(' (')[0]} at threshold t = {t:.2f}")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for c, label, val in [
        (k1, "Cost per week", f"${m['cost_week']:,.0f}"),
        (k2, "Expedited / week", f"{m['expedited'] / 5:.0f}"),
        (k3, "Lates missed / week", f"{m['fn'] / 5:.1f}"),
        (k4, "Accuracy", f"{m['accuracy']:.1%}"),
        (k5, "Precision", "-" if np.isnan(m['precision']) else f"{m['precision']:.1%}"),
        (k6, "Recall", f"{m['recall']:.1%}"),
    ]:
        c.markdown(f"<div class='small-label'>{label}</div><div class='big-number'>{val}</div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.05, 1.4])
    with left:
        st.markdown("**Confusion matrix (5 weeks of shipments)**")
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='cm-cell' style='background:{ORANGE}'><div class='cm-title'>Caught it</div><div class='cm-num'>{m['tp']}</div><div class='cm-sub'>late, expedited (TP) · paid ${fee:,} each</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='cm-cell' style='background:{SKY}'><div class='cm-title'>Wasted expedite</div><div class='cm-num'>{m['fp']}</div><div class='cm-sub'>on time, expedited (FP) · paid ${fee:,} each</div></div>", unsafe_allow_html=True)
        st.write("")
        c3, c4 = st.columns(2)
        c3.markdown(f"<div class='cm-cell' style='background:#9B2D00'><div class='cm-title'>Missed late shipment</div><div class='cm-num'>{m['fn']}</div><div class='cm-sub'>late, not expedited (FN) · paid ${penalty:,} each</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='cm-cell' style='background:{BLUE}'><div class='cm-title'>Correct wait</div><div class='cm-num'>{m['tn']}</div><div class='cm-sub'>on time, not expedited (TN) · $0</div></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f"<div class='idea'><b>Cost per week</b> = {fee:,} × ({m['tp']} + {m['fp']}) + {penalty:,} × {m['fn']} = ${m['cost_total']:,.0f} over 5 weeks "
            f"→ <b>${m['cost_week']:,.0f} / week</b></div>", unsafe_allow_html=True)
        st.write("")
        comp = pd.DataFrame({
            "Policy": ["Never expedite", "Expedite everything", f"Your threshold t = {t:.2f}", f"Formula t* = fee/penalty = {t_star:.2f}", f"Best threshold on this data = {bt:.2f}"],
            "Cost / week": [m_never["cost_week"], m_all["cost_week"], m["cost_week"], m_star["cost_week"], bc],
            "Expedited / week": [0, per_week, m["expedited"] / 5, m_star["expedited"] / 5, metrics(y, score, bt, fee, penalty, per_week)["expedited"] / 5],
            "Lates missed / week": [m_never["fn"] / 5, 0, m["fn"] / 5, m_star["fn"] / 5, metrics(y, score, bt, fee, penalty, per_week)["fn"] / 5],
        })
        st.dataframe(comp.style.format({"Cost / week": "${:,.0f}", "Expedited / week": "{:.0f}", "Lates missed / week": "{:.1f}"}), hide_index=True, width="stretch")

    with right:
        cc = pd.DataFrame({"t": grid, "cost": costs})
        base = alt.Chart(cc).mark_line(color=BLUE, strokeWidth=2.5).encode(
            x=alt.X("t:Q", title="Threshold t  (expedite if probability ≥ t)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("cost:Q", title="Expected cost per week ($)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("t:Q", format=".2f"), alt.Tooltip("cost:Q", format="$,.0f")],
        )
        here = alt.Chart(pd.DataFrame({"t": [t], "cost": [m["cost_week"]]})).mark_point(size=260, filled=True, color=ORANGE).encode(x="t:Q", y="cost:Q")
        star = alt.Chart(pd.DataFrame({"t": [t_star]})).mark_rule(color=SKY, strokeDash=[6, 4], strokeWidth=2).encode(x="t:Q")
        star_txt = alt.Chart(pd.DataFrame({"t": [t_star], "label": [f"formula t* = {t_star:.2f}"]})).mark_text(align="left", dx=6, dy=-6, color=SKY, fontWeight="bold").encode(x="t:Q", y=alt.value(12), text="label:N")
        never = alt.Chart(pd.DataFrame({"cost": [m_never["cost_week"]]})).mark_rule(color=GRAY, strokeDash=[2, 3]).encode(y="cost:Q")
        never_txt = alt.Chart(pd.DataFrame({"t": [0.55], "cost": [m_never["cost_week"]], "label": ["never expedite"]})).mark_text(align="left", dy=-8, color=GRAY).encode(x="t:Q", y="cost:Q", text="label:N")
        st.altair_chart((base + never + never_txt + star + star_txt + here).properties(height=330, title="Cost per week for every threshold  (orange dot = you)"), width="stretch")

        # butterfly histogram: on-time above, late below, each with its own scale
        bins = np.arange(0, 1.0001, 0.05)
        c_on, _ = np.histogram(score[y == 0], bins=bins)
        c_late, _ = np.histogram(score[y == 1], bins=bins)
        hist = pd.DataFrame({"bin_start": bins[:-1], "bin_end": bins[1:], "On time": c_on, "Late": c_late})
        rule = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(color=NAVY, strokeWidth=3).encode(x="t:Q")
        top = alt.Chart(hist).mark_bar(color=BLUE).encode(
            x=alt.X("bin_start:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labels=False, ticks=False)), x2="bin_end:Q",
            y=alt.Y("On time:Q", title="On time (first bar cut at 350)", scale=alt.Scale(domain=[0, 350], clamp=True)), y2=alt.datum(0),
            tooltip=[alt.Tooltip("bin_start:Q", title="probability from", format=".2f"), alt.Tooltip("On time:Q", title="on-time shipments")],
        ).properties(height=130, title="Where the line falls: right of the line = expedite")
        bottom = alt.Chart(hist).mark_bar(color=ORANGE).encode(
            x=alt.X("bin_start:Q", title="Model probability of arriving late", scale=alt.Scale(domain=[0, 1])), x2="bin_end:Q",
            y=alt.Y("Late:Q", title="Late", scale=alt.Scale(reverse=True)), y2=alt.datum(0),
            tooltip=[alt.Tooltip("bin_start:Q", title="probability from", format=".2f"), alt.Tooltip("Late:Q", title="late shipments")],
        ).properties(height=110)
        st.altair_chart(alt.vconcat(top + rule, bottom + rule, spacing=2).resolve_scale(x="shared"), width="stretch")
        st.caption("Blue bars (up): on-time shipments. Orange bars (down): late shipments. Right of the line you pay the fee; left of the line you risk the penalty.")

    st.info("**The model ranks. The business decides where to cut.**  Try: set the scenario to *Air-freight expedite* and leave t at 0.10 - the old threshold becomes very expensive.")

# ------------------------------------------------------------- tab 2 -------
with tab2:
    st.subheader("Round 1 - ten shipments arriving before the White Out game")
    st.write(f"Expedite fee **\\${fee:,}** per shipment you expedite. Late penalty **\\${penalty:,}** per late shipment you did not expedite. Pick the shipments to expedite, then reveal.")
    r1 = d.set_index("ship_id").loc[ROUND1].reset_index()
    r1.insert(0, "Shipment", [f"S{i+1}" for i in range(10)])
    show = r1[["Shipment", "origin", "miles", "carrier", "mode", "lead_time_days", "snow_forecast", "supplier_on_time_pct"]].rename(columns={
        "origin": "Origin", "miles": "Miles", "carrier": "Carrier", "mode": "Mode", "lead_time_days": "Lead time (days)",
        "snow_forecast": "Snow forecast", "supplier_on_time_pct": "Supplier on-time %"})
    show["Snow forecast"] = show["Snow forecast"].map({0: "no", 1: "yes"})
    st.dataframe(show, hide_index=True, width="stretch")
    picks = st.multiselect("Which shipments do you expedite?", list(show["Shipment"]), default=[])
    reveal = st.toggle("Reveal what actually happened")
    if reveal:
        r1["Expedited"] = r1["Shipment"].isin(picks)
        r1["Actually late"] = r1["late"] == 1
        def cell(row):
            if row["Expedited"] and row["Actually late"]: return "Caught it (TP)"
            if row["Expedited"] and not row["Actually late"]: return "Wasted expedite (FP)"
            if (not row["Expedited"]) and row["Actually late"]: return "Missed late shipment (FN)"
            return "Correct wait (TN)"
        r1["Result"] = r1.apply(cell, axis=1)
        tp = int((r1["Expedited"] & r1["Actually late"]).sum()); fp = int((r1["Expedited"] & ~r1["Actually late"]).sum())
        fn = int((~r1["Expedited"] & r1["Actually late"]).sum()); tn = 10 - tp - fp - fn
        cost = fee * (tp + fp) + penalty * fn
        a, b, c_, e = st.columns(4)
        a.metric("Your cost", f"${cost:,}")
        b.metric("Never expedite", f"${penalty * 3:,}")
        c_.metric("Expedite all ten", f"${fee * 10:,}")
        e.metric("Perfect information", f"${fee * 3:,}")
        st.write(f"Caught {tp} · Wasted {fp} · Missed {fn} · Correct waits {tn}")
        out = r1[["Shipment", "origin", "carrier", "mode", "Result", "p_vendor", "p_homemade"]].rename(columns={"origin": "Origin", "carrier": "Carrier", "mode": "Mode", "p_vendor": "Vendor model p(late)", "p_homemade": "Homemade model p(late)"})
        st.dataframe(out.style.map(lambda v: "background-color:#FBE3D0" if isinstance(v, str) and v.startswith("Missed") else ("background-color:#E3F1FA" if isinstance(v, str) and v.startswith("Wasted") else ""), subset=["Result"]).format({"Vendor model p(late)": "{:.2f}", "Homemade model p(late)": "{:.2f}"}), hide_index=True, width="stretch")
        st.markdown("**Now let the model decide.** What does each threshold do on these ten?")
        rows = []
        for tt in (0.50, 0.25, t_star):
            mm = metrics(r1.late.values, r1.p_vendor.values, tt, fee, penalty, per_week=10)
            rows.append({"Threshold": f"{tt:.2f}" + ("  (formula t*)" if abs(tt - t_star) < 1e-9 else ""), "Expedited": mm["expedited"], "Caught": mm["tp"], "Wasted": mm["fp"], "Missed": mm["fn"], "Cost": mm["cost_total"]})
        st.dataframe(pd.DataFrame(rows).style.format({"Cost": "${:,.0f}"}), hide_index=True, width="stretch")

# ------------------------------------------------------------- tab 3 -------
with tab3:
    st.subheader("Two models, one dock")
    V = d.p_vendor.values; Hm = d.p_homemade.values
    a1, a2 = st.columns(2)
    a1.metric("Vendor model - AUC", f"{auc(y, V):.3f}", help="Gradient boosting, 40 features")
    a2.metric("Homemade model - AUC", f"{auc(y, Hm):.3f}", help="Logistic regression, 5 features + the carrier's 'behind schedule' alert")
    fv, tv = roc_curve(y, V); fh, th = roc_curve(y, Hm)
    roc = pd.concat([pd.DataFrame({"fpr": fv, "tpr": tv, "Model": "Vendor"}), pd.DataFrame({"fpr": fh, "tpr": th, "Model": "Homemade"})])
    roc_chart = alt.Chart(roc).mark_line(strokeWidth=2.5).encode(
        x=alt.X("fpr:Q", title="False positive rate (on-time shipments you expedite)"),
        y=alt.Y("tpr:Q", title="Recall (late shipments you catch)"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Vendor", "Homemade"], range=[BLUE, ORANGE]), legend=alt.Legend(orient="bottom-right", title=None)),
        tooltip=["Model:N", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#C4CAD1", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    b1, b2 = st.columns([1.25, 1])
    with b1:
        st.altair_chart((diag + roc_chart).properties(height=360, title="ROC: the whole curve"), width="stretch")
    with b2:
        st.markdown("**No cap: pick the threshold from the costs**")
        rows = []
        for name, s in [("Vendor", V), ("Homemade", Hm)]:
            mf = metrics(y, s, t_star, fee, penalty, per_week); bt_, bc_ = best_threshold(y, s, fee, penalty, per_week)
            rows.append({"Model": name, f"Cost/wk at t*={t_star:.2f}": mf["cost_week"], "Best t (data)": bt_, "Cost/wk at best t": bc_})
        st.dataframe(pd.DataFrame(rows).style.format({f"Cost/wk at t*={t_star:.2f}": "${:,.0f}", "Cost/wk at best t": "${:,.0f}", "Best t (data)": "{:.2f}"}), hide_index=True, width="stretch")
        st.markdown("**With a cap: you can only expedite this many per week**")
        cap = st.slider("Expedites allowed per week", 5, 120, 20, 5)
        k = int(cap * 5)
        rows = []
        for name, s in [("Vendor", V), ("Homemade", Hm)]:
            mk = metrics_topk(y, s, k, fee, penalty, per_week)
            rows.append({"Model": name, "Lates caught (of 169)": mk["tp"], "Precision": mk["precision"], "Cost/wk": mk["cost_week"], "Implied t": mk["cutoff"]})
        st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.0%}", "Cost/wk": "${:,.0f}", "Implied t": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Under a small cap only the far-left of the ROC curve matters. AUC averages over the whole curve, including thresholds you will never use.")

# ------------------------------------------------------------- tab 4 -------
with tab4:
    st.subheader("Are the probabilities honest? (calibration)")
    st.write("The formula t* = fee / penalty only works if a '0.10' really means 'late one time in ten'. Group shipments by their predicted probability and check.")
    cal = pd.concat([calibration_table(y, d.p_vendor.values).assign(Model="Vendor"), calibration_table(y, d.p_homemade.values).assign(Model="Homemade")])
    line = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=2.5).encode(
        x=alt.X("avg_score:Q", title="Average predicted probability in the bin", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("late_rate:Q", title="Share that actually arrived late", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Vendor", "Homemade"], range=[BLUE, ORANGE])),
        tooltip=["Model:N", "bin:N", "n:Q", alt.Tooltip("avg_score:Q", format=".2f"), alt.Tooltip("late_rate:Q", format=".2f")],
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#C4CAD1", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    st.altair_chart((diag + line).properties(height=380, title="Points on the dashed line = honest probabilities"), width="stretch")
    st.dataframe(cal.style.format({"avg_score": "{:.3f}", "late_rate": "{:.3f}"}), hide_index=True, width="stretch")
    st.caption("If a model is not calibrated, either recalibrate it (Platt scaling, isotonic regression) or pick the threshold from the cost curve on real data instead of from the formula.")

# ------------------------------------------------------------- tab 5 -------
with tab5:
    st.subheader("The data")
    st.write("2,000 inbound shipments to the Bellefonte DC over 5 weeks. Made-up data with a fixed seed, so every number in class can be reproduced.")
    st.dataframe(d, hide_index=True, width="stretch", height=420)
    st.download_button("Download shipments.csv", d.to_csv(index=False).encode(), "shipments.csv", "text/csv")
    st.markdown("""
| Column | Meaning |
|---|---|
| `origin`, `miles` | where the truck starts and road miles to Bellefonte, PA |
| `carrier`, `mode` | made-up carriers; full truckload, LTL or parcel |
| `lead_time_days` | days between pickup and the promised arrival |
| `snow_forecast` | 1 if snow is forecast on the route (hello, Seven Mountains on US-322) |
| `supplier_on_time_pct` | the supplier's on-time history |
| `carrier_alert` | 1 if the carrier's system flagged the truck as behind schedule |
| `late` | 1 if the shipment actually arrived late |
| `p_vendor`, `p_homemade` | each model's probability that the shipment arrives late |
""")

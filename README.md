# Expedite or Wait? — the classroom threshold tool

A small Streamlit app for the 90-minute lesson *"Expedite or Wait? Why a more accurate model can make worse decisions."*
Students move a decision threshold and watch the confusion matrix, the metrics and the weekly cost change.

## Files

| File | What it is |
|---|---|
| `app.py` | the Streamlit app (five tabs: threshold, Round 1, model showdown, calibration, data) |
| `expedite_data.py` | generates the 2,000-shipment dataset from a fixed seed and holds the metric functions |
| `shipments.csv` | the same data as a file, for students who want it |
| `requirements.txt` | Python packages |

## Run it on your laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Share that window on Zoom if the public link ever fails.

## Put it online for free (Streamlit Community Cloud)

1. Create a public GitHub repository and upload the four files.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**, pick the repository and `app.py`, click **Deploy**.
3. You get a link like `https://your-app.streamlit.app`. Test it on a phone. Put the link and a QR code on slides 25 and 34.
4. Free apps go to sleep after a few days without visitors. Open the link ten minutes before class to wake it up.

## Class scenarios built in

| Scenario | Expedite fee | Late penalty | Formula threshold t* = fee / penalty |
|---|---|---|---|
| Game-day week | $150 | $1,500 | 0.10 |
| Normal week | $150 | $600 | 0.25 |
| Air-freight expedite | $600 | $900 | 0.67 |
| Custom | you type it | you type it | computed live |

## The two models

* **Vendor model** — "gradient boosting, 40 features". AUC 0.86. Wins when there is no cap on expedites.
* **Homemade model** — "logistic regression, 5 features + the carrier's *behind schedule* alert". AUC 0.76. Wins when the dock can only expedite about 20 shipments a week, because its first picks are almost never wrong.

Every number in the slide deck and the lecture notes comes from `expedite_data.py` with seed 814, so the tool and the slides always agree.

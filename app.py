# app.py — Auto-Download aus LimeSurvey + Auswertung (Fortlaufend & Kompakt) + PDF
import io
import json
import zipfile
import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ------------------ Konfiguration ------------------
# Erst versuchen, Streamlit-Secrets zu laden
if "LS_API_URL" in st.secrets:
    API_URL = st.secrets["LS_API_URL"].strip()
    LS_USER = st.secrets["LS_USER"].strip()
    LS_PASSWORD = st.secrets["LS_PASSWORD"].strip()
    LS_SID = int(st.secrets["LS_SURVEY_ID"])
else:
    # Lokaler Fallback: .env-Datei verwenden
    load_dotenv()
    API_URL = os.getenv("LS_API_URL", "").strip()
    LS_USER = os.getenv("LS_USER", "").strip()
    LS_PASSWORD = os.getenv("LS_PASSWORD", "").strip()
    LS_SID = int(os.getenv("LS_SURVEY_ID", "0"))

OUTFILE = Path(f"survey_{LS_SID}_responses.json")

# kurze X-Achsen-Labels
EVAL_LABELS = {
    "Evaluation[SQ001]": "Relevanz Themen",
    "Evaluation[SQ002]": "Konstruktiver Austausch",
    "Evaluation[SQ003]": "Lernerfolg",
    "Evaluation[SQ004]": "Entlastung",
    "Evaluation[SQ005]": "Moderation",
    "Evaluation[SQ006]": "Umsetzbarkeit Alltag",
    "Evaluation[SQ007]": "Gesamtwert"
}
EVAL_COLS = list(EVAL_LABELS.keys())

# bevorzugte Reihenfolge der Stufen in Plot/Report
STAGE_ORDER = [
    "Sitzungen 1–3",
    "Sitzungen 4–6",
    "Eintages-Kompaktkurs",
    "Nachtreffen",
]

st.set_page_config(page_title="Evaluation Dashboard", layout="centered")
st.title("📊 Evaluation Lehrer*innen Coachinggruppen")

# ------------------ LimeSurvey RPC Helper ------------------
def _rpc(method: str, params: list):
    """JSON-RPC Call"""
    payload = {"method": method, "params": params, "id": 1}
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("Zeitüberschreitung beim Kontakt zum LimeSurvey-Server.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP-Fehler beim RPC-Aufruf: {e}")
    except ValueError:
        raise RuntimeError("Antwort des Servers war kein gültiges JSON.")

    if data.get("error"):
        raise RuntimeError(f"RPC-Fehler: {data['error']}")
    return data.get("result")

def _decode_export_to_json(result_obj):
    """export_responses Ergebnis (Base64/ZIP/JSON) in Python-Objekt dekodieren"""
    if isinstance(result_obj, (dict, list)):
        return result_obj
    if isinstance(result_obj, str):
        try:
            raw = base64.b64decode(result_obj, validate=True)
        except Exception:
            return json.loads(result_obj)
    else:
        raise RuntimeError("export_responses: Unerwartetes Format")

    bio = io.BytesIO(raw)
    if zipfile.is_zipfile(bio):
        with zipfile.ZipFile(bio) as zf:
            for nm in zf.namelist():
                if nm.lower().endswith(".json"):
                    return json.loads(zf.read(nm).decode("utf-8", errors="replace"))
        raise RuntimeError("ZIP ohne JSON-Datei")
    return json.loads(raw.decode("utf-8", errors="replace"))

# ------------------ Immer frischer Abruf aus LimeSurvey ------------------
def fetch_latest_json():
    """Holt aktuelle Antworten, speichert OUTFILE und gibt Python-Objekt zurück."""
    if not (API_URL and LS_USER and LS_PASSWORD and LS_SID):
        raise RuntimeError(
            "Fehlende Zugangsdaten. Bitte LS_API_URL, LS_USER, LS_PASSWORD, LS_SURVEY_ID "
            "in Streamlit Secrets oder in einer lokalen .env setzen."
        )

    sess = _rpc("get_session_key", [LS_USER, LS_PASSWORD])
    try:
        # WICHTIG: "full" statt "code" für Antworttexte
        res = _rpc("export_responses", [sess, LS_SID, "json", None, "all", "full"])
    finally:
        try:
            _rpc("release_session_key", [sess])
        except Exception:
            pass

    data = _decode_export_to_json(res)
    OUTFILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

# ------------------ Daten laden ------------------
try:
    with st.spinner("Lade aktuelle Daten aus LimeSurvey…"):
        raw_json = fetch_latest_json()
    st.caption(f"Quelle: Live-Export Survey {LS_SID} → gespeichert als {OUTFILE.name}")
except Exception as e:
    st.error(f"Fehler beim Abrufen der Daten: {e}")
    st.stop()

def normalize_records(obj):
    if isinstance(obj, list):
        recs = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("responses"), list):
            recs = obj["responses"]
        elif isinstance(obj.get("responses"), dict):
            recs = list(obj["responses"].values())
        elif isinstance(obj.get("data"), list):
            recs = obj["data"]
        else:
            recs = []
    else:
        recs = []

    norm = []
    for item in recs:
        if isinstance(item, dict) and isinstance(item.get("answers"), dict):
            flat = {k: v for k, v in item.items() if k != "answers"}
            flat.update(item["answers"])
            norm.append(flat)
        else:
            norm.append(item)
    return norm

records = normalize_records(raw_json)
df = pd.DataFrame.from_records(records)
if df.empty:
    st.warning("Keine Antworten vorhanden.")
    st.stop()

df = df.rename(columns=lambda x: str(x).strip())

for col in ["Kennung", "WieVielSitzungenFort", "WelcheSitzungenKompa", "Anmerkung"]:
    if col not in df.columns:
        alt = [c for c in df.columns if c.lower() == col.lower()]
        if alt:
            df.rename(columns={alt[0]: col}, inplace=True)
        else:
            df[col] = np.nan

for i in range(1, 8):
    b = f"Evaluation[SQ00{i}]"
    u = f"Evaluation_SQ00{i}"
    if b not in df.columns and u in df.columns:
        df[b] = df[u]
    if b not in df.columns:
        df[b] = pd.NA

for c in EVAL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["WieVielSitzungenFort"] = pd.to_numeric(df["WieVielSitzungenFort"], errors="coerce")
df["WelcheSitzungenKompa"] = pd.to_numeric(df["WelcheSitzungenKompa"], errors="coerce")

# ------------------ Erhebungszeitpunkt bestimmen ------------------
def row_stage_label(row):
    wv = row.get("WieVielSitzungenFort")
    if pd.notna(wv) and int(wv) in (1, 2):
        return "Sitzungen 1–3" if int(wv) == 1 else "Sitzungen 4–6"
    wk = row.get("WelcheSitzungenKompa")
    if pd.notna(wk) and int(wk) in (1, 2):
        return "Eintages-Kompaktkurs" if int(wk) == 1 else "Nachtreffen"
    return None

df["Erhebungszeitpunkt"] = df.apply(row_stage_label, axis=1)

# ------------------ Kennung auswählen ------------------
kennungen = sorted(df["Kennung"].dropna().astype(str).str.strip().unique())
if not kennungen:
    st.warning("Keine Kennungen in den Daten.")
    st.stop()

selected = st.selectbox(
    "Kennung auswählen (einzeln)",
    options=kennungen,
    index=0,
    help="Es wird genau eine Kennung betrachtet (keine Kumulierung)."
)

df_sel = df[df["Kennung"].astype(str).str.strip() == str(selected)]
if df_sel.empty:
    st.warning("Keine Antworten für die gewählte Kennung.")
    st.stop()

# ------------------ Mittelwerte & n je Stufe ------------------
series_means = {}
series_n = {}
for label in STAGE_ORDER:
    sub = df_sel[df_sel["Erhebungszeitpunkt"] == label]
    if not sub.empty:
        series_means[label] = sub[EVAL_COLS].mean(numeric_only=True)
        series_n[label] = int(sub.shape[0])

means_overall = None
n_overall = None
if not series_means:
    means_overall = df_sel[EVAL_COLS].mean(numeric_only=True)
    n_overall = int(df_sel.shape[0])

# ------------------ Diagramm ------------------
fig, ax = plt.subplots(figsize=(10, 5.8))
x = np.arange(len(EVAL_COLS))

if series_means:
    labels_present = [lab for lab in STAGE_ORDER if lab in series_means]
    k = len(labels_present)
    width = min(0.8 / max(k, 1), 0.35)
    offsets = np.linspace(-(k-1)/2, (k-1)/2, k) * width

    color_map = {
        "Sitzungen 1–3": "#6baed6",
        "Eintages-Kompaktkurs": "#6baed6",
        "Sitzungen 4–6": "#2171b5",
        "Nachtreffen": "#2171b5"
    }

    for j, lab in enumerate(labels_present):
        vals = [series_means[lab].get(c, np.nan) for c in EVAL_COLS]
        ax.bar(
            x + offsets[j],
            vals,
            width,
            label=f"{lab} (n={series_n[lab]})",
            color=color_map.get(lab, "#1f77b4")
        )
else:
    width = 0.5
    vals = [means_overall.get(c, np.nan) for c in EVAL_COLS]
    ax.bar(x, vals, width, label=f"Gesamt (n={n_overall})", color="#2171b5")

ax.set_ylabel("Mittelwert")
ax.set_title(f"Evaluation der Coachinggruppe ({selected})" + (" – nach Erhebungszeitpunkten" if series_means else ""))
ax.set_xticks(x)
ax.set_xticklabels([EVAL_LABELS[c] for c in EVAL_COLS], rotation=20, ha="right")
ax.legend(loc="best")
plt.tight_layout()
st.pyplot(fig)

# ------------------ Info n ------------------
if series_means:
    info = " · ".join([f"{lab}: n={series_n[lab]}" for lab in [lab for lab in STAGE_ORDER if lab in series_means]])
    st.info(f"Teilnahmen: {info}")
else:
    st.info(f"Teilnahmen gesamt (ohne Stufenangabe): n={n_overall}")

# ------------------ Anmerkungen ------------------
st.subheader("💬 Anmerkungen")
ann = df_sel.copy()
ann["Anmerkung"] = ann["Anmerkung"].astype(str).str.strip()
ann = ann[ann["Anmerkung"].notna() & (ann["Anmerkung"] != "")]

def notes_for(label):
    return ann[ann["Erhebungszeitpunkt"] == label]["Anmerkung"].tolist()

if series_means:
    for lab in [lab for lab in STAGE_ORDER if lab in series_means]:
        st.markdown(f"**{lab}**")
        notes = notes_for(lab)
        if notes:
            for a in notes:
                st.write(f"- {a}")
        else:
            st.write("Keine Anmerkungen.")
else:
    st.markdown("**Gesamt**")
    notes = ann["Anmerkung"].tolist()
    if notes:
        for a in notes:
            st.write(f"- {a}")
    else:
        st.write("Keine Anmerkungen.")

# ------------------ PDF-Export ------------------
def build_pdf_bytes(kennung, fig, series_means, series_n, means_overall=None, n_overall=None):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Bericht zur Kennung {kennung}", styles["Title"]))
    story.append(Spacer(1, 0.15*inch))

    if series_means:
        info = " · ".join([f"{lab}: <b>n={series_n[lab]}</b>" for lab in [lab for lab in STAGE_ORDER if lab in series_means]])
        story.append(Paragraph(f"Teilnahmen: {info}", styles["Normal"]))
    else:
        story.append(Paragraph(f"Teilnahmen gesamt (ohne Stufenangabe): <b>n={n_overall}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=200)
    img_buf.seek(0)
    img_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img_tmp.write(img_buf.getvalue())
    img_tmp.flush()
    story.append(Image(img_tmp.name, width=6.7*inch, height=4.2*inch))
    story.append(Spacer(1, 0.25*inch))

    if series_means:
        for lab in [lab for lab in STAGE_ORDER if lab in series_means]:
            story.append(Paragraph(f"<b>{lab}</b>", styles["Heading2"]))
            notes = notes_for(lab)
            if notes:
                for a in notes:
                    story.append(Paragraph(f"- {a}", styles["Normal"]))
            else:
                story.append(Paragraph("Keine Anmerkungen.", styles["Normal"]))
            story.append(Spacer(1, 0.15*inch))
    else:
        story.append(Paragraph("<b>Gesamt</b>", styles["Heading2"]))
        notes = ann["Anmerkung"].tolist()
        if notes:
            for a in notes:
                story.append(Paragraph(f"- {a}", styles["Normal"]))
        else:
            story.append(Paragraph("Keine Anmerkungen.", styles["Normal"]))

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4)
    doc.build(story)
    return pdf_buf.getvalue()

pdf_bytes = build_pdf_bytes(
    kennung=selected,
    fig=fig,
    series_means=series_means,
    series_n=series_n,
    means_overall=means_overall,
    n_overall=n_overall
)

st.download_button(
    label="📄 Bericht als PDF herunterladen",
    data=pdf_bytes,
    file_name=f"Bericht_{selected}.pdf",
    mime="application/pdf",
    use_container_width=True
)

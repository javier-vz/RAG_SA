#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lenguas Indígenas de Sudamérica — Explorer
==========================================
Dark mode · Mobile-first · GraphRAG · Anthropic Claude
"""

import os
import re
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Lenguas Indígenas · Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { background: #0d1117 !important; font-family: 'Inter', sans-serif !important; }

.block-container {
    max-width: 700px !important;
    margin: 0 auto !important;
    padding: 0 1.25rem 4rem 1.25rem !important;
    background: transparent !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

p, span, label, div, li { color: #e6edf3; }
h1,h2,h3,h4,h5,h6 { color: #e6edf3 !important; }
code { background: #21262d !important; color: #79c0ff !important; border-radius: 4px; padding: 0 4px; }

.app-header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 0.8rem 1.25rem;
    margin: 0 -1.25rem 1.75rem -1.25rem;
    display: flex;
    align-items: center;
}
.app-header-title { font-size: 1.05rem; font-weight: 700; color: #e6edf3; margin: 0; }
.app-header-sub   { font-size: 0.72rem; color: #8b949e; margin-left: auto; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    border: 1px solid #30363d !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 6px !important;
    border: none !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* INPUTS */
.stTextInput > div > div > input {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2ea043 !important;
    box-shadow: 0 0 0 2px rgba(46,160,67,.2) !important;
}
.stTextInput > div > div > input::placeholder { color: #8b949e !important; }
.stTextInput label { color: #8b949e !important; font-size: 0.8rem !important; }

/* SELECTBOX */
.stSelectbox > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
.stSelectbox label { color: #8b949e !important; font-size: 0.8rem !important; }
.stSelectbox > div > div > div { color: #e6edf3 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="popover"] > div > div
    { background: #21262d !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
[data-baseweb="menu"] { background: #21262d !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
[data-baseweb="menu"] ul, [data-baseweb="menu"] li { background: #21262d !important; color: #e6edf3 !important; }
[role="listbox"], [role="listbox"] > div, [role="listbox"] li, [role="option"]
    { background: #21262d !important; color: #e6edf3 !important; font-size: 0.88rem !important; }
[role="option"]:hover { background: #30363d !important; }
[aria-selected="true"][role="option"] { background: #30363d !important; color: #e6edf3 !important; }
[data-baseweb="popover"] * { color: #e6edf3 !important; }
[data-baseweb="popover"] div { background: #21262d !important; }

/* RADIO */
.stRadio > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    gap: 1.5rem !important;
}
.stRadio label { color: #adbac7 !important; font-size: 0.85rem !important; }

/* BOTONES */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    transition: all .15s !important;
}
.stButton > button[kind="primary"] {
    background: #2ea043 !important;
    color: #fff !important;
    width: 100% !important;
    padding: 0.6rem !important;
    box-shadow: 0 2px 8px rgba(46,160,67,.3) !important;
}
.stButton > button[kind="primary"]:hover { background: #238636 !important; }
.stButton > button[kind="secondary"] {
    background: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    width: 100% !important;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
    font-size: 0.83rem !important;
}
.streamlit-expanderContent {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
}

/* METRICS */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stMetricValue"] { color: #2ea043 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }

/* DIVIDER */
hr { border-color: #30363d !important; margin: 1rem 0 !important; }

/* CHAT */
.msg-user {
    background: rgba(31,111,235,.1);
    border: 1px solid rgba(56,139,253,.25);
    border-radius: 10px 10px 4px 10px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
}
.msg-user .lbl { font-size:0.68rem; font-weight:700; color:#58a6ff; text-transform:uppercase; letter-spacing:.5px; margin-bottom:5px; }
.msg-user .txt { font-size:0.88rem; color:#e6edf3; }

.msg-bot {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #2ea043;
    border-radius: 4px 10px 10px 10px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
}
.msg-bot .lbl { font-size:0.68rem; font-weight:700; color:#2ea043; text-transform:uppercase; letter-spacing:.5px; margin-bottom:5px; }
.msg-bot .txt { font-size:0.88rem; color:#adbac7; line-height:1.7; }

/* FICHA / CARD */
.lang-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.lang-card h4 { font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.5px; color:#2ea043; margin:0 0 0.5rem 0; }
.lang-card p  { font-size:0.84rem; line-height:1.72; color:#adbac7; margin:0; }
.lang-card p+p { margin-top:0.4rem; }

/* ABOUT */
.about-sec {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
}
.about-sec h4 { font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.5px; color:#2ea043; margin:0 0 0.6rem 0; }
.about-sec p  { font-size:0.84rem; line-height:1.72; color:#adbac7; margin:0; }
.about-sec p+p { margin-top:0.5rem; }

.chips { display:flex; flex-wrap:wrap; gap:0.3rem; margin-top:0.6rem; }
.chip  { font-size:0.67rem; padding:0.18rem 0.5rem; border-radius:4px;
    font-family:monospace; border:1px solid #30363d;
    background:rgba(255,255,255,.04); color:#adbac7; }
.chip.b{ background:rgba(56,139,253,.1); border-color:rgba(56,139,253,.3); color:#79c0ff; }
.chip.g{ background:rgba(46,160,67,.1);  border-color:rgba(46,160,67,.3);  color:#2ea043; }

/* FOOTER */
.tfooter { text-align:center; padding:1.5rem 0 0.5rem; border-top:1px solid #30363d; margin-top:2rem; }
.tfooter p { font-size:0.72rem; color:#8b949e; margin:0.2rem 0; }
.empty-state { text-align:center; padding:2.5rem 1rem; color:#8b949e; font-size:0.85rem; }

/* VITALITY BADGE */
.badge { display:inline-block; font-size:0.65rem; font-weight:700;
    padding:0.15rem 0.5rem; border-radius:20px; margin-left:0.4rem; }
.badge-extinct  { background:rgba(248,81,73,.15); color:#f85149; border:1px solid rgba(248,81,73,.3); }
.badge-critical { background:rgba(210,153,34,.15); color:#d49922; border:1px solid rgba(210,153,34,.3); }
.badge-alive    { background:rgba(46,160,67,.15);  color:#2ea043; border:1px solid rgba(46,160,67,.3); }
</style>
""", unsafe_allow_html=True)


#
# CARGAR GRAFO (cached)
#

@st.cache_resource
def cargar_grafo(ttl_path: str):
    """Parsea el TTL y devuelve entidades ABox indexadas."""
    content = Path(ttl_path).read_text(encoding="utf-8")
    blocks  = re.split(r"\n(?=ex:[a-z])", content)

    entities = {}
    family_members = defaultdict(list)

    def extr(block, prop):
        m = re.search(rf'ex:{prop}\s+"""(.*?)"""', block, re.DOTALL)
        if m: return m.group(1).strip()
        m = re.search(rf'ex:{prop}\s+"([^"]+)"', block)
        if m: return m.group(1).strip()
        return ""

    def extr_comment(block):
        m = re.search(r'rdfs:comment\s+"""(.*?)"""', block, re.DOTALL)
        if m: return m.group(1).strip()
        m = re.search(r'rdfs:comment\s+"((?:[^\\"]|\\.)+)"', block)
        if m: return m.group(1).strip()
        return ""

    def extr_families(block):
        fams = re.findall(
            r"ex:belongsToFamily\s+((?:ex:\w+(?:,\s*\n?\s*)?)+)", block, re.DOTALL)
        if not fams: return []
        return re.findall(r"ex:(\w+)", fams[0])

    for b in blocks:
        m = re.match(r"ex:(\w+)", b.strip())
        if not m: continue
        eid = m.group(1)
        is_lang = "a ex:Language" in b
        is_fam  = "a ex:Family"   in b
        if not (is_lang or is_fam): continue

        families = extr_families(b)
        wiki_text = extr(b, "wikipedia_summary").lower()

        is_historical = (
        "extinct" in wiki_text or
        "was an extinct" in wiki_text or
        "there were" in wiki_text
        )
        
        entities[eid] = {
            "type":     "Language" if is_lang else "Family",
            "label":    extr(b, "label") or eid,
            "comment":  extr_comment(b),
            "iso":      extr(b, "iso"),
            "countries": extr(b, "wikidata_countries"),
            "speakers":  extr(b, "wikidata_speakers"),
            "wiki":      extr(b, "wikipedia_summary"),
            "families":  families,
            "members":   [],
            "is_historical": is_historical,
        }
        for fid in families:
            family_members[fid].append(eid)

    for fid, lids in family_members.items():
        if fid in entities:
            entities[fid]["members"] = lids

    return entities


@st.cache_resource
def cargar_motor(ttl_path: str, matrix_path: str, idmap_path: str):
    """Carga embeddings y construye índice lexical."""
    from sentence_transformers import SentenceTransformer

    matrix = np.load(matrix_path).astype(np.float32)
    with open(idmap_path, "rb") as f:
        id_map = pickle.load(f)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Índice lexical
    lex_index = defaultdict(list)

    def normalize(text):
        text = text.lower()
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            text = text.replace(a, b)
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if len(w) > 2]

    entities = cargar_grafo(ttl_path)
    for eid in id_map:
        if eid not in entities: continue
        ent = entities[eid]
        tokens = set()
        tokens.update(normalize(ent["label"]))
        tokens.update(normalize(ent["comment"]))
        tokens.update(normalize(ent.get("countries", "")))
        tokens.update(normalize(ent.get("wiki", "")[:200]))
        for tok in tokens:
            lex_index[tok].append(eid)

    return {
        "model":     model,
        "matrix":    matrix,
        "id_map":    id_map,
        "lex_index": lex_index,
        "normalize": normalize,
    }


#
# RETRIEVAL
#


PAISES = {
    "perú": "Perú", "peru": "Perú", "brasil": "Brasil", "brazil": "Brasil",
    "colombia": "Colombia", "bolivia": "Bolivia", "venezuela": "Venezuela",
    "ecuador": "Ecuador", "argentina": "Argentina", "chile": "Chile",
    "paraguay": "Paraguay", "guyana": "Guyana", "surinam": "Surinam",
    "panama": "Panamá", "panamá": "Panamá", "uruguay": "Uruguay",
}
FAMILIAS_QUERY = {
    "tupian": "tupi1275", "tupi": "tupi1275",
    "arawak": "araw1281", "arawakan": "araw1281",
    "quechua": "quec1387", "cariban": "cari1283", "carib": "cari1283",
    "tucanoan": "tuca1253", "panoan": "pano1256",
    "yanomamic": "yano1268", "yanomami": "yano1268",
    "guaicuruan": "guai1249", "guarani": "tupi1276",
    "chibchan": "chib1249", "aymaran": "ayma1253",
    "araucanian": "arau1255", "chocoan": "choc1280",
}

def detect_country(query):
    q = query.lower()
    for kw, norm in PAISES.items():
        if kw in q: return norm
    return None

def detect_family(query):
    q = query.lower()
    for kw, fid in FAMILIAS_QUERY.items():
        if kw in q: return fid
    return None


PAISES = {
    "perú": "Perú", "peru": "Perú", "brasil": "Brasil", "brazil": "Brasil",
    "colombia": "Colombia", "bolivia": "Bolivia", "venezuela": "Venezuela",
    "ecuador": "Ecuador", "argentina": "Argentina", "chile": "Chile",
    "paraguay": "Paraguay", "guyana": "Guyana", "surinam": "Surinam",
    "panama": "Panamá", "panamá": "Panamá", "uruguay": "Uruguay",
}

FAMILIAS_QUERY = {
    "tupian": "tupi1275", "tupi": "tupi1275",
    "arawak": "araw1281", "arawakan": "araw1281",
    "quechua": "quec1387", "cariban": "cari1283", "carib": "cari1283",
    "tucanoan": "tuca1253", "panoan": "pano1256",
    "yanomamic": "yano1268", "yanomami": "yano1268",
    "guaicuruan": "guai1249", "guarani": "tupi1276",
    "chibchan": "chib1249", "aymaran": "ayma1253",
    "araucanian": "arau1255", "chocoan": "choc1280",
}


def detect_country(query):
    q = query.lower()
    for kw, norm in PAISES.items():
        if kw in q:
            return norm
    return None


def detect_family(query):
    q = query.lower()
    for kw, fid in FAMILIAS_QUERY.items():
        if kw in q:
            return fid
    return None



PAISES = {
    "peru": "Perú", "perú": "Perú", "brasil": "Brasil", "brazil": "Brasil",
    "colombia": "Colombia", "bolivia": "Bolivia", "venezuela": "Venezuela",
    "ecuador": "Ecuador", "argentina": "Argentina", "chile": "Chile",
    "paraguay": "Paraguay", "guyana": "Guyana", "surinam": "Surinam",
    "panama": "Panamá", "panamá": "Panamá", "uruguay": "Uruguay",
}

FAMILIAS_QUERY = {
    "tupian": "tupi1275", "tupi": "tupi1275",
    "arawak": "araw1281", "arawakan": "araw1281",
    "quechua": "quec1387", "cariban": "cari1283", "carib": "cari1283",
    "tucanoan": "tuca1253", "panoan": "pano1256",
    "yanomamic": "yano1268", "yanomami": "yano1268",
    "guaicuruan": "guai1249", "guarani": "tupi1276",
    "chibchan": "chib1249", "aymaran": "ayma1253",
    "araucanian": "arau1255", "chocoan": "choc1280",
}


def detect_country(query):
    q = query.lower()
    for kw, norm in PAISES.items():
        if kw in q:
            return norm
    return None


def detect_family(query):
    q = query.lower()
    for kw, fid in FAMILIAS_QUERY.items():
        if kw in q:
            return fid
    return None


PAISES = {
    "peru": "Perú", "perú": "Perú", "brasil": "Brasil", "brazil": "Brasil",
    "colombia": "Colombia", "bolivia": "Bolivia", "venezuela": "Venezuela",
    "ecuador": "Ecuador", "argentina": "Argentina", "chile": "Chile",
    "paraguay": "Paraguay", "guyana": "Guyana", "surinam": "Surinam",
    "panama": "Panamá", "panamá": "Panamá", "uruguay": "Uruguay",
}

FAMILIAS_QUERY = {
    "tupian": "tupi1275", "tupi": "tupi1275",
    "arawak": "araw1281", "arawakan": "araw1281",
    "quechua": "quec1387", "cariban": "cari1283", "carib": "cari1283",
    "tucanoan": "tuca1253", "panoan": "pano1256",
    "yanomamic": "yano1268", "yanomami": "yano1268",
    "guaicuruan": "guai1249", "guarani": "tupi1276",
    "chibchan": "chib1249", "aymaran": "ayma1253",
    "araucanian": "arau1255", "chocoan": "choc1280",
}


def detect_country(query):
    q = query.lower()
    for kw, norm in PAISES.items():
        if kw in q:
            return norm
    return None


def detect_family(query):
    q = query.lower()
    for kw, fid in FAMILIAS_QUERY.items():
        if kw in q:
            return fid
    return None


def retrieve(query: str, motor: dict, top_k: int = 10, alpha: float = 0.6):
    matrix    = motor["matrix"]
    id_map    = motor["id_map"]
    model     = motor["model"]
    lex_index = motor["lex_index"]
    normalize = motor["normalize"]
    entities  = st.session_state.get("_entities", {})

    country_filter = detect_country(query)
    family_filter  = detect_family(query)

    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    sims  = (matrix @ q_emb.T).flatten()
    sem   = {id_map[i]: float(sims[i]) for i in range(len(id_map))}

    q_tokens = set(normalize(query))
    lex_raw = defaultdict(float)
    for tok in q_tokens:
        for eid in lex_index.get(tok, []):
            lex_raw[eid] += 1.0

    q_lower = query.lower()
    for eid in id_map:
        ent = entities.get(eid)
        if ent and q_lower in ent["label"].lower():
            lex_raw[eid] = lex_raw.get(eid, 0) + 3.0

    if country_filter:
        for eid, ent in entities.items():
            if ent.get("type") == "Language" and ent.get("countries") == country_filter:
                lex_raw[eid] = lex_raw.get(eid, 0) + 5.0

    if family_filter:
        lex_raw[family_filter] = lex_raw.get(family_filter, 0) + 6.0
        for mid in entities.get(family_filter, {}).get("members", []):
            lex_raw[mid] = lex_raw.get(mid, 0) + 4.0

    max_lex = max(lex_raw.values()) if lex_raw else 1.0
    lex = {eid: s / max_lex for eid, s in lex_raw.items()}
    combined = {
        eid: alpha * sem.get(eid, 0.0) + (1 - alpha) * lex.get(eid, 0.0)
        for eid in set(sem) | set(lex)
    }

    if country_filter:
        st.session_state["_query_country"] = country_filter
    else:
        st.session_state.pop("_query_country", None)
    if family_filter:
        st.session_state["_query_family"] = family_filter
    else:
        st.session_state.pop("_query_family", None)

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

RELATION_MAP = {
    "belongsToFamily":      "belongs to family",
    "semanticallySimilarTo": "semantically similar to",
    "wikidata_countries":   "spoken in",
    "wikidata_speakers":    "speakers",
}




def build_context(entity_ids: list, entities: dict, max_chars: int = 2500) -> str:
    country_filter = st.session_state.get("_query_country")
    family_filter  = st.session_state.get("_query_family")

    if country_filter:
        langs = [
            (eid, ent) for eid, ent in entities.items()
            if ent.get("type") == "Language"
            and ent.get("countries") == country_filter
            and ent.get("speakers") not in ("0", "", None)
            and not ent.get("is_historical", False)
    ]
        langs.sort(key=lambda x: -(int(x[1]["speakers"]) if (x[1].get("speakers") or "").isdigit() else 0))
        n = len(langs)
        names = ", ".join(e["label"] for _, e in langs[:50])
        if n > 50:
            names += " ... y " + str(n - 50) + " mas"
        details = []
        for _, ent in langs[:8]:
            spk = ent.get("speakers") or "sin dato"
            fams = [entities[f]["label"] for f in ent["families"][:2] if f in entities]
            fam_str = " (" + ", ".join(fams) + ")" if fams else ""
            details.append("  - " + ent["label"] + fam_str + ": " + str(spk) + " hablantes")
        return (
            "[COUNTRY QUERY] Lenguas asociadas a " + country_filter + "\n\n"
            "Complete list: " + names + "\n\n"
            "Most spoken (with data):\n" + "\n".join(details)
        )

    if family_filter and family_filter in entities:
        fam = entities[family_filter]
        members = fam.get("members", [])
        details = []
        for mid in members[:25]:
            m = entities.get(mid)
            if not m:
                continue
            line = "  - " + m["label"]
            if m.get("countries"):
                line += " (" + m["countries"] + ")"
            if m.get("speakers"):
                line += ": " + m["speakers"] + " hablantes"
            details.append(line)
        extra = "\n  ... y " + str(len(members) - 25) + " mas" if len(members) > 25 else ""
        return (
            "[Family] " + fam["label"] + " - " + str(len(members)) + " member languages\n\n"
            + "\n".join(details) + extra
        )

    parts, total = [], 0
    for eid in entity_ids[:5]:
        ent = entities.get(eid)
        if not ent:
            continue
        lines = ["[" + ent["type"] + "] " + ent["label"]]
        if ent["comment"]:
            lines.append("  " + ent["comment"][:300])
        elif ent["wiki"]:
            lines.append("  " + ent["wiki"][:200])
        if ent["iso"]:
            lines.append("  ISO 639-3: " + ent["iso"])
        if ent["countries"]:
            lines.append("  Countries: " + ent["countries"])
        if ent.get("speakers"):
            spk = ent["speakers"]
            lines.append("  Speakers: " + ("extinct" if spk == "0" else spk))
        fam_names = [entities[f]["label"] for f in ent["families"][:3] if f in entities]
        if fam_names:
            lines.append("  belongs to family: " + ", ".join(fam_names))
        if ent["type"] == "Family" and ent["members"]:
            mnames = [entities[m]["label"] for m in ent["members"][:12] if m in entities]
            if mnames:
                lines.append("  member languages: " + ", ".join(mnames))
        block = "\n".join(lines)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "No relevant information found."


# GENERATION
#

SYSTEM_PROMPT = """You are a specialist assistant in South American indigenous linguistics.

Answer ONLY using the provided knowledge graph context.

Rules:
- Respond in the same language as the question.
- Be clear, brief, and natural.
- Write for a broad audience.

IMPORTANT:
- Distinguish the type of question.

IF the question asks "cuántas":
- Give ONLY an approximate number.
- Do NOT list languages.
- Use expressions like: "aproximadamente", "alrededor de".

IF the question asks "qué lenguas":
- Mention several representative languages.
- Do NOT try to give a full list.
- Do NOT emphasize the total number.

General:
- Do not present graph totals as exact real-world facts.
- If information may be incomplete, speak in approximate terms.
- Do not mention the context or system.
"""

def generate(question: str, context: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Knowledge graph context:\n\n{context}\n\n-----\nQuestion: {question}\n\nAnswer based ONLY on the context above."}
        ],
        max_tokens=400,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


#
# VITALITY BADGE
#

def vitality_badge(speakers: str) -> str:
    if not speakers: return ""
    if speakers == "0": return '<span class="badge badge-extinct">Extinta</span>'
    try:
        n = int(speakers.replace(",", ""))
        if n < 100:   return '<span class="badge badge-critical">Crítica</span>'
        if n < 1000:  return '<span class="badge badge-critical">Amenazada</span>'
        return '<span class="badge badge-alive">Activa</span>'
    except:
        return ""


#
# ESTADO
#

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

def agregar_mensaje(tipo, texto):
    st.session_state.mensajes.append({"tipo": tipo, "texto": texto})

def limpiar_chat():
    st.session_state.mensajes = []


#
# PREGUNTAS SUGERIDAS
#

PREGUNTAS = [
    "¿Qué lenguas pertenecen a la familia Tupiana?",
    "¿Qué es el Pirahã?",
    "¿Dónde se habla el Mapudungun?",
    "¿Qué lenguas están extintas en Argentina?",
    "¿Cuántos hablantes tiene el Quechua de Cusco?",
    "¿Qué lenguas de señas existen en Sudamérica?",
    "¿Qué es el Yanomamö?",
    "¿Qué lenguas pertenecen a la familia Arawak?",
    "¿Qué es el Wayuu?",
    "¿Qué lenguas se hablan en Bolivia?",
    "¿Qué es el Aymara?",
    "¿Cuáles son las lenguas más habladas de la región?",
    "¿Qué lenguas están en peligro crítico?",
    "¿Qué es el Asháninka?",
    "¿Cuántas lenguas tupí-guaraní existen?",
    "¿Qué es el Quechua de Ayacucho?",
]

#
# ARCHIVOS
#

TTL_PATH    = "grafo_enriquecido.ttl"
MATRIX_PATH = "embed_matrix_v2.npy"
IDMAP_PATH  = "id_map_v2.pkl"

#
# HEADER
#

st.markdown("""
<div class="app-header">
    <div class="app-header-title">🗺 Lenguas Indígenas · Sudamérica</div>
    <div class="app-header-sub">GraphRAG Explorer</div>
</div>
""", unsafe_allow_html=True)

#
# CARGAR DATOS
#

files_ok = all(Path(p).exists() for p in [TTL_PATH, MATRIX_PATH, IDMAP_PATH])

if not files_ok:
    missing = [p for p in [TTL_PATH, MATRIX_PATH, IDMAP_PATH] if not Path(p).exists()]
    st.error(f"Archivos faltantes: {', '.join(missing)}")
    st.info("Asegúrate de tener en el mismo directorio: `grafo_enriquecido.ttl`, `embed_matrix_v2.npy`, `id_map_v2.pkl`")
    st.stop()

with st.spinner("Cargando grafo y embeddings…"):
    entities = cargar_grafo(TTL_PATH)
    st.session_state["_entities"] = entities
    motor    = cargar_motor(TTL_PATH, MATRIX_PATH, IDMAP_PATH)

#
# TABS
#

st.markdown("""
<p style="font-size:.82rem;color:#8b949e;line-height:1.65;margin-bottom:.75rem">
Explora el grafo de conocimiento de lenguas indígenas de Sudamérica.
Consulta en lenguaje natural, explora lenguas por familia o país, o revisa
la arquitectura del sistema.
</p>
""", unsafe_allow_html=True)

tab_chat, tab_explorar, tab_about = st.tabs(["💬  Consultas", "🔍  Explorar", "ℹ️  Acerca de"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONSULTAS (Chat)
# ═════════════════════════════════════════════════════════════════════════════

with tab_chat:

    modo = st.radio(
        "modo", ["📋 Preguntas sugeridas", "✍️ Pregunta libre"],
        horizontal=True, label_visibility="collapsed"
    )

    pregunta = ""
    if modo == "📋 Preguntas sugeridas":
        pregunta = st.selectbox(
            "pregunta", options=[""] + PREGUNTAS,
            format_func=lambda x: "Selecciona una pregunta…" if x == "" else x,
            label_visibility="collapsed"
        )
    else:
        pregunta = st.text_input(
            "pregunta",
            placeholder="¿Qué lenguas pertenecen a la familia Arawak?",
            label_visibility="collapsed"
        )
        with st.expander("💡 Ver ejemplos"):
            for p in PREGUNTAS[:6]:
                st.markdown(f"- {p}")

    if st.button("✨  Preguntar", type="primary", use_container_width=True):
        if pregunta:
            agregar_mensaje("user", pregunta)
            with st.spinner("Consultando grafo de conocimiento…"):
                try:
                    results  = retrieve(pregunta, motor)
                    ids      = [eid for eid, _ in results]
                    context  = build_context(ids, entities)
                    respuesta = generate(pregunta, context)
                    agregar_mensaje("bot", respuesta)
                except Exception as e:
                    agregar_mensaje("bot", f"Error: {e}")
            st.rerun()

    if st.session_state.mensajes:
        st.markdown("---")
        for msg in reversed(st.session_state.mensajes):
            if msg["tipo"] == "user":
                st.markdown(
                    f'<div class="msg-user"><div class="lbl">Tú</div>'
                    f'<div class="txt">{msg["texto"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="msg-bot"><div class="lbl">Asistente</div>'
                    f'<div class="txt">{msg["texto"]}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown(" ")
        if st.button("🗑️  Limpiar historial", use_container_width=True):
            limpiar_chat()
            st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">Selecciona o escribe una pregunta para comenzar</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORAR
# ═════════════════════════════════════════════════════════════════════════════

with tab_explorar:

    # -- Filtros
    col1, col2 = st.columns(2)

    with col1:
        tipo_sel = st.selectbox(
            "Tipo", ["Language", "Family"],
            label_visibility="visible"
        )

    # Collect unique countries for filter
    all_countries = sorted(set(
        ent["countries"]
        for ent in entities.values()
        if ent["type"] == "Language" and ent["countries"]
    ))

    with col2:
        if tipo_sel == "Language":
            pais_sel = st.selectbox(
                "País", ["Todos"] + all_countries,
                label_visibility="visible"
            )
        else:
            pais_sel = "Todos"

    busqueda = st.text_input(
        "Buscar", placeholder="Filtrar por nombre…",
        label_visibility="collapsed"
    )

    # -- Filtrar entidades
    filtered = [
        (eid, ent) for eid, ent in entities.items()
        if ent["type"] == tipo_sel
        and (pais_sel == "Todos" or ent.get("countries") == pais_sel)
        and (not busqueda or busqueda.lower() in ent["label"].lower())
    ]
    filtered.sort(key=lambda x: x[1]["label"])

    st.markdown(
        f'<p style="font-size:0.78rem;color:#8b949e;margin-bottom:0.75rem">'
        f'{len(filtered)} entidades encontradas</p>',
        unsafe_allow_html=True
    )

    # -- Mostrar cards
    for eid, ent in filtered[:50]:
        badge = vitality_badge(ent["speakers"]) if ent["type"] == "Language" else ""

        # Build subtitle line
        meta_parts = []
        if ent["iso"]:       meta_parts.append(f"ISO: {ent['iso']}")
        if ent["countries"]: meta_parts.append(ent["countries"])
        if ent["speakers"] and ent["speakers"] != "0":
            meta_parts.append(f"{ent['speakers']} hablantes")
        if ent["type"] == "Family":
            meta_parts.append(f"{len(ent['members'])} lenguas")

        meta = " · ".join(meta_parts) if meta_parts else ""

        # Short description
        desc = ent["comment"] or ent["wiki"]
        desc = desc[:220] + "…" if len(desc) > 220 else desc

        # Families
        fam_labels = [entities[f]["label"] for f in ent["families"][:3] if f in entities]
        fam_str = ", ".join(fam_labels) if fam_labels else ""

        st.markdown(f"""
<div class="lang-card">
  <h4>{ent['label']}{badge}</h4>
  <p style="font-size:0.72rem;color:#8b949e;margin-bottom:0.5rem">{meta}</p>
  {'<p>' + desc + '</p>' if desc else ''}
  {'<p style="font-size:0.75rem;color:#8b949e;margin-top:0.4rem">Familia: ' + fam_str + '</p>' if fam_str else ''}
</div>
""", unsafe_allow_html=True)

    if len(filtered) > 50:
        st.markdown(
            f'<p style="font-size:0.78rem;color:#8b949e;text-align:center">'
            f'Mostrando 50 de {len(filtered)}. Usa el filtro para reducir.</p>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACERCA DE
# ═════════════════════════════════════════════════════════════════════════════

with tab_about:

    n_langs = sum(1 for e in entities.values() if e["type"] == "Language")
    n_fams  = sum(1 for e in entities.values() if e["type"] == "Family")
    n_comment = sum(1 for e in entities.values() if e["type"] == "Language" and e["comment"])

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Lenguas", n_langs)
    with c2: st.metric("Familias", n_fams)
    with c3: st.metric("Con rdfs:comment", n_comment)

    st.markdown("""
<div class="about-sec">
<h4>🗺 El proyecto</h4>
<p>Sistema GraphRAG para explorar el grafo de conocimiento de lenguas indígenas
de Sudamérica. El grafo integra datos de Glottolog, Wikidata, Wikipedia y PHOIBLE
sobre 552 lenguas y 409 familias lingüísticas, con énfasis en lenguas en peligro.</p>
<p>El pipeline combina búsqueda híbrida (semántica + léxica) sobre el grafo RDF
con generación de lenguaje natural condicionada al contexto recuperado,
siguiendo la arquitectura del paper <em>Situated Knowledge in Knowledge Graphs</em>.</p>
</div>
<div class="about-sec">
<h4>⚙️ Stack técnico</h4>
<div class="chips">
  <span class="chip g">GraphRAG</span>
  <span class="chip g">Knowledge Graph</span>
  <span class="chip g">Lenguas Indígenas</span>
  <span class="chip b">RDF / Turtle</span>
  <span class="chip b">RDFS / OWL</span>
  <span class="chip b">FAISS</span>
  <span class="chip">Groq llama-3.3-70b</span>
  <span class="chip">sentence-transformers</span>
  <span class="chip">paraphrase-multilingual-MiniLM</span>
  <span class="chip">rdflib</span>
  <span class="chip">Streamlit</span>
</div>
</div>
<div class="about-sec">
<h4>📐 Arquitectura del pipeline</h4>
<p>
<strong>Fase 1</strong> — Enriquecimiento: generación de <code>rdfs:comment</code>
semántico por lengua usando Claude (491/552 cubiertas).<br>
<strong>Fase 2</strong> — Índice FAISS: embeddings de 961 entidades ABox
(Language + Family) con <code>paraphrase-multilingual-MiniLM-L12-v2</code>.<br>
<strong>Fase 3</strong> — Retrieval híbrido: semántico (α=0.6) + léxico (α=0.4),
top-10 candidatos, solo ABox (sin contaminación TBox).<br>
<strong>Fase 4</strong> — Generación: LLM condicionado al contexto del grafo,
sin alucinaciones.
</p>
</div>
<div class="about-sec">
<h4>📊 Cobertura geográfica</h4>
<p>Brasil · Perú · Colombia · Bolivia · Venezuela · Ecuador · Argentina ·
Chile · Paraguay · Guyana · Surinam · Uruguay · Trinidad y Tobago · Panamá</p>
</div>
""", unsafe_allow_html=True)


#
# FOOTER
#

st.markdown("""
<div class="tfooter">
    <p>🗺️ <b>Lenguas Indígenas · Sudamérica</b> · GraphRAG Explorer</p>
    <p>Knowledge Graph + RAG · Glottolog · Wikidata · Wikipedia · PHOIBLE</p>
    <p style="color:#30363d;font-size:0.65rem;margin-top:0.3rem;">
        RDF/Turtle · Groq llama-3.3-70b · paraphrase-multilingual-MiniLM-L12-v2 · FAISS
    </p>
</div>
""", unsafe_allow_html=True)

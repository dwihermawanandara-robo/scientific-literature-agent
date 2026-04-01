import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from extractor import extract_text_from_pdf
from prompts import (
    SYSTEM_PROMPT,
    COMPARE_PROMPT,
    RELATED_WORK_PROMPT,
    RECOMMENDATION_PROMPT,
)


# =========================
# Setup
# =========================
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key) if api_key else None

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.small-muted {
    color: #9aa0a6;
    font-size: 0.92rem;
}
.section-title {
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Research Assistant Agent")
st.caption(
    "Upload maksimal 2 paper PDF untuk ekstraksi, AI summary, comparison, related work draft, recommendation panel, confidence tracking, dan history panel."
)


# =========================
# Session State
# =========================
if "basic_infos" not in st.session_state:
    st.session_state.basic_infos = {}

if "paper_summaries" not in st.session_state:
    st.session_state.paper_summaries = {}

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}

if "related_work_results" not in st.session_state:
    st.session_state.related_work_results = {}

if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = {}


# =========================
# Constants
# =========================
SUMMARY_CORE_FIELDS = [
    "title",
    "research_problem",
    "method",
    "dataset",
    "metrics",
    "main_results",
    "novelty",
    "limitations",
]

SUMMARY_EVIDENCE_FIELDS = [
    "evidence_problem",
    "evidence_method",
    "evidence_results",
    "evidence_novelty",
]

COMPARISON_CORE_FIELDS = [
    "key_difference",
    "paper_1_strength",
    "paper_2_strength",
    "practical_takeaway",
    "method_gap",
    "dataset_gap",
    "evaluation_gap",
    "implementation_gap",
    "future_direction",
]

RELATED_WORK_FIELDS = [
    "related_work_paragraph",
    "positioning_statement",
]

RECOMMENDATION_FIELDS = [
    "more_practical_paper",
    "more_novel_paper",
    "better_baseline_paper",
    "better_for_implementation_reference",
    "better_for_research_inspiration",
    "recommendation_reasoning",
]


# =========================
# Generic Helper functions
# =========================
def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def make_safe_name(name: str, max_len: int = 80) -> str:
    name = Path(name).stem
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return name[:max_len] if name else "file"


def save_json(prefix: str, file_name: str, data: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prefix in {"comparison", "related_work", "recommendation"} and "__VS__" in file_name:
        parts = file_name.split("__VS__")
        left = make_safe_name(parts[0], 35)
        right = make_safe_name(parts[1], 35)
        safe_name = f"{left}_VS_{right}"
    else:
        safe_name = make_safe_name(file_name, 70)

    output_path = OUTPUT_DIR / f"{prefix}_{safe_name}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(output_path)


def normalize_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if clean:
            lines.append(clean)
    return lines


def is_missing_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"", "-", "not clearly stated", "not available", "unknown", "none"}

    if isinstance(value, list):
        return len(value) == 0

    return False


def compute_field_completeness(data: dict, fields: list[str]) -> tuple[int, int, float]:
    filled = 0
    total = len(fields)

    for field in fields:
        if field in data and not is_missing_value(data.get(field)):
            filled += 1

    ratio = filled / total if total > 0 else 0.0
    return filled, total, ratio


def map_ratio_to_label(ratio: float) -> str:
    if ratio >= 0.85:
        return "High"
    if ratio >= 0.60:
        return "Medium"
    return "Low"


def compute_summary_quality(summary_data: dict) -> dict:
    filled_core, total_core, core_ratio = compute_field_completeness(summary_data, SUMMARY_CORE_FIELDS)
    filled_evidence, total_evidence, evidence_ratio = compute_field_completeness(summary_data, SUMMARY_EVIDENCE_FIELDS)

    score = (0.65 * core_ratio) + (0.35 * evidence_ratio)
    confidence_label = map_ratio_to_label(score)

    return {
        "filled_core": filled_core,
        "total_core": total_core,
        "core_ratio": core_ratio,
        "filled_evidence": filled_evidence,
        "total_evidence": total_evidence,
        "evidence_ratio": evidence_ratio,
        "confidence_score": score,
        "confidence_label": confidence_label,
    }


def compute_comparison_quality(comparison_data: dict) -> dict:
    filled, total, ratio = compute_field_completeness(comparison_data, COMPARISON_CORE_FIELDS)
    return {
        "filled": filled,
        "total": total,
        "ratio": ratio,
        "label": map_ratio_to_label(ratio),
    }


def compute_related_work_quality(related_work_data: dict) -> dict:
    filled, total, ratio = compute_field_completeness(related_work_data, RELATED_WORK_FIELDS)
    return {
        "filled": filled,
        "total": total,
        "ratio": ratio,
        "label": map_ratio_to_label(ratio),
    }


def compute_recommendation_quality(recommendation_data: dict) -> dict:
    filled, total, ratio = compute_field_completeness(recommendation_data, RECOMMENDATION_FIELDS)
    return {
        "filled": filled,
        "total": total,
        "ratio": ratio,
        "label": map_ratio_to_label(ratio),
    }


# =========================
# PDF Heuristic Extraction
# =========================
def is_journal_header_line(line: str) -> bool:
    lower = line.lower()
    journal_keywords = [
        "ieee transactions",
        "ieee access",
        "international journal",
        "transactions on",
        "vol.",
        "no.",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    return any(keyword in lower for keyword in journal_keywords)


def is_metadata_line(line: str) -> bool:
    lower = line.lower()

    metadata_keywords = [
        "received",
        "accepted",
        "date of publication",
        "date of current version",
        "digital object identifier",
        "doi",
        "issn",
        "www.",
        "http",
        "copyright",
        "corresponding author",
        "e-issn",
        "p-issn",
        "impact factor",
        "page ",
        "open access",
    ]

    if any(keyword in lower for keyword in metadata_keywords):
        return True

    if is_journal_header_line(line):
        return True

    return False


def is_affiliation_line(line: str) -> bool:
    lower = line.lower()

    affiliation_keywords = [
        "university",
        "institute",
        "department",
        "school",
        "laboratory",
        "faculty",
        "college",
        "china",
        "pakistan",
        "india",
        "indonesia",
        "malaysia",
        "email",
        "dept.",
        "campus",
    ]

    if any(keyword in lower for keyword in affiliation_keywords):
        return True

    if re.match(r"^\d+\s*", line):
        return True

    return False


def is_author_line(line: str) -> bool:
    if len(line) < 8:
        return False

    lower = line.lower()

    if re.search(r"\b(member|student|prof|professor|dr\.)\b", lower):
        return True

    if line.count(",") >= 2 and re.search(r"[A-Z]", line):
        return True

    words = line.split()
    upper_words = [w for w in words if len(w) > 1 and w.isupper()]
    if len(upper_words) >= 3:
        return True

    return False


def is_possible_title_line(line: str) -> bool:
    if is_metadata_line(line) or is_affiliation_line(line) or is_author_line(line):
        return False

    if len(line) < 8 or len(line) > 160:
        return False

    lower = line.lower()
    bad_keywords = ["abstract", "keywords", "index terms", "introduction"]

    if any(keyword in lower for keyword in bad_keywords):
        return False

    return True


def clean_title_text(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"\b\d+\b", "", title).strip()
    title = re.sub(r"\s{2,}", " ", title)
    return title


def guess_title(text: str) -> str:
    lines = normalize_lines(text[:7000])

    abstract_idx = None
    for i, line in enumerate(lines):
        if re.search(r"(?i)^abstract\b", line):
            abstract_idx = i
            break

    if abstract_idx is None:
        abstract_idx = min(len(lines), 40)

    candidate_region = lines[:abstract_idx]
    filtered = [line for line in candidate_region if not is_metadata_line(line)]

    title_lines = []
    started = False

    for line in filtered:
        if not started:
            if is_possible_title_line(line):
                started = True
                title_lines.append(line)
        else:
            if is_possible_title_line(line):
                title_lines.append(line)
                if len(title_lines) >= 4:
                    break
            else:
                break

    if not title_lines:
        fallback = []
        for line in filtered[:12]:
            if is_possible_title_line(line):
                fallback.append(line)
        title_lines = fallback[:3]

    title = " ".join(title_lines).strip()
    title = clean_title_text(title)

    if not title:
        return "Title not found clearly."

    return title


def extract_authors(text: str) -> str:
    lines = normalize_lines(text[:8000])

    abstract_idx = None
    for i, line in enumerate(lines):
        if re.search(r"(?i)^abstract\b", line):
            abstract_idx = i
            break

    if abstract_idx is None:
        abstract_idx = min(len(lines), 40)

    candidate_region = lines[:abstract_idx]

    authors = []
    for line in candidate_region:
        if is_metadata_line(line):
            continue
        if is_affiliation_line(line):
            continue
        if is_author_line(line):
            authors.append(line)

    if not authors:
        return "Authors not found clearly."

    return " ".join(authors[:2])


def extract_abstract(text: str) -> str:
    text_short = text[:20000]

    patterns = [
        r"(?is)\babstract\b\s*[-—–:]*\s*(.+?)(?=\bkeywords?\b|\bindex terms\b|\bintroduction\b|\b1\.\s*introduction\b|\bi\.\s*introduction\b)",
        r"(?is)\babstract\b\s*[-—–:]*\s*(.+?)(?=\n\s*\n)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_short)
        if match:
            abstract = match.group(1)
            abstract = re.sub(r"\s+", " ", abstract).strip()
            abstract = re.sub(r"^(abstract\s*[-—–:]*)", "", abstract, flags=re.I).strip()
            return abstract[:1200]

    return "Abstract not found clearly."


def clean_preview_text(text: str) -> str:
    lines = normalize_lines(text[:3500])

    cleaned = []
    for line in lines:
        if is_metadata_line(line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned[:20])


def extract_basic_info(text: str, file_name: str) -> dict:
    return {
        "file_name": file_name,
        "character_count": len(text),
        "title_guess": guess_title(text),
        "authors_guess": extract_authors(text),
        "abstract_preview": extract_abstract(text),
    }


# =========================
# AI Functions
# =========================
def summarize_paper_with_ai(paper_text: str) -> dict:
    truncated_text = paper_text[:15000]

    user_prompt = f"""
Read the following scientific paper text and return a valid JSON object
using the exact schema from the system instruction.

Paper text:
{truncated_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = response.choices[0].message.content or "{}"
    cleaned_output = clean_json_text(raw_output)
    return json.loads(cleaned_output)


def compare_two_summaries_with_ai(summary1: dict, summary2: dict) -> dict:
    user_prompt = f"""
Compare these two structured paper summaries.

Paper 1 summary:
{json.dumps(summary1, ensure_ascii=False, indent=2)}

Paper 2 summary:
{json.dumps(summary2, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": COMPARE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = response.choices[0].message.content or "{}"
    cleaned_output = clean_json_text(raw_output)
    return json.loads(cleaned_output)


def generate_related_work_with_ai(summary1: dict, summary2: dict, comparison: dict) -> dict:
    user_prompt = f"""
Write a related work draft based on these materials.

Paper 1 summary:
{json.dumps(summary1, ensure_ascii=False, indent=2)}

Paper 2 summary:
{json.dumps(summary2, ensure_ascii=False, indent=2)}

Comparison result:
{json.dumps(comparison, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RELATED_WORK_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = response.choices[0].message.content or "{}"
    cleaned_output = clean_json_text(raw_output)
    return json.loads(cleaned_output)


def generate_recommendation_with_ai(summary1: dict, summary2: dict, comparison: dict) -> dict:
    user_prompt = f"""
Recommend which paper is better for different purposes based on these materials.

Paper 1 summary:
{json.dumps(summary1, ensure_ascii=False, indent=2)}

Paper 2 summary:
{json.dumps(summary2, ensure_ascii=False, indent=2)}

Comparison result:
{json.dumps(comparison, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RECOMMENDATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = response.choices[0].message.content or "{}"
    cleaned_output = clean_json_text(raw_output)
    return json.loads(cleaned_output)


# =========================
# Report Builders
# =========================
def build_comparison_dataframe(summary1: dict, summary2: dict) -> pd.DataFrame:
    def metrics_to_text(metrics):
        if isinstance(metrics, list):
            return ", ".join(metrics) if metrics else "Not clearly stated"
        return str(metrics)

    rows = [
        ["Title", summary1.get("title", "-"), summary2.get("title", "-")],
        ["Method", summary1.get("method", "-"), summary2.get("method", "-")],
        ["Dataset", summary1.get("dataset", "-"), summary2.get("dataset", "-")],
        ["Metrics", metrics_to_text(summary1.get("metrics", [])), metrics_to_text(summary2.get("metrics", []))],
        ["Main Results", summary1.get("main_results", "-"), summary2.get("main_results", "-")],
        ["Novelty", summary1.get("novelty", "-"), summary2.get("novelty", "-")],
        ["Limitations", summary1.get("limitations", "-"), summary2.get("limitations", "-")],
    ]

    return pd.DataFrame(rows, columns=["Aspect", "Paper 1", "Paper 2"])


def summary_to_markdown(summary_data: dict, paper_label: str, file_name: str) -> str:
    metrics = summary_data.get("metrics", [])
    metrics_text = ", ".join(metrics) if isinstance(metrics, list) and metrics else "Not clearly stated"

    return f"""# {paper_label} Summary

**File:** {file_name}

## Title
{summary_data.get("title", "-")}

## Research Problem
{summary_data.get("research_problem", "-")}

## Method
{summary_data.get("method", "-")}

## Dataset
{summary_data.get("dataset", "-")}

## Metrics
{metrics_text}

## Main Results
{summary_data.get("main_results", "-")}

## Novelty
{summary_data.get("novelty", "-")}

## Limitations
{summary_data.get("limitations", "-")}

## Evidence Snippets
- Problem: {summary_data.get("evidence_problem", "-")}
- Method: {summary_data.get("evidence_method", "-")}
- Results: {summary_data.get("evidence_results", "-")}
- Novelty: {summary_data.get("evidence_novelty", "-")}
"""


def comparison_to_markdown(comparison: dict, file1: str, file2: str) -> str:
    return f"""# Comparison Report

**Paper 1:** {file1}
**Paper 2:** {file2}

## Key Difference
{comparison.get("key_difference", "-")}

## Paper 1 Strength
{comparison.get("paper_1_strength", "-")}

## Paper 2 Strength
{comparison.get("paper_2_strength", "-")}

## Practical Takeaway
{comparison.get("practical_takeaway", "-")}

## Structured Research Gap
### Method Gap
{comparison.get("method_gap", "-")}

### Dataset Gap
{comparison.get("dataset_gap", "-")}

### Evaluation Gap
{comparison.get("evaluation_gap", "-")}

### Implementation Gap
{comparison.get("implementation_gap", "-")}

## Future Direction
{comparison.get("future_direction", "-")}
"""


def related_work_to_markdown(related_work: dict, file1: str, file2: str) -> str:
    return f"""# Related Work Draft

**Paper 1:** {file1}
**Paper 2:** {file2}

## Related Work Paragraph
{related_work.get("related_work_paragraph", "-")}

## Positioning Statement
{related_work.get("positioning_statement", "-")}
"""


def recommendation_to_markdown(recommendation: dict, file1: str, file2: str) -> str:
    return f"""# Recommendation Panel

**Paper 1:** {file1}
**Paper 2:** {file2}

## More Practical Paper
{recommendation.get("more_practical_paper", "-")}

## More Novel Paper
{recommendation.get("more_novel_paper", "-")}

## Better Baseline Paper
{recommendation.get("better_baseline_paper", "-")}

## Better for Implementation Reference
{recommendation.get("better_for_implementation_reference", "-")}

## Better for Research Inspiration
{recommendation.get("better_for_research_inspiration", "-")}

## Recommendation Reasoning
{recommendation.get("recommendation_reasoning", "-")}
"""


# =========================
# UI Display Helpers
# =========================
def display_basic_info_card(info: dict, paper_label: str):
    with st.container(border=True):
        st.markdown(f"### {paper_label}")
        st.caption(info.get("file_name", "-"))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Character Count**")
            st.write(f"{info.get('character_count', 0):,}")

            st.markdown("**Authors Guess**")
            st.write(info.get("authors_guess", "-"))

        with col2:
            st.markdown("**Title Guess**")
            st.info(info.get("title_guess", "-"))

        st.markdown("**Abstract Preview**")
        st.text_area(
            label=f"abstract_{paper_label}",
            value=info.get("abstract_preview", "-"),
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )


def format_metrics(metrics):
    if isinstance(metrics, list):
        return ", ".join(metrics) if metrics else "Not clearly stated"
    return str(metrics)


def display_summary_card(summary_data: dict, paper_label: str, file_name: str):
    quality = compute_summary_quality(summary_data)

    with st.container(border=True):
        st.markdown(f"### {paper_label}")
        st.caption(file_name)

        q1, q2 = st.columns(2)
        with q1:
            st.metric("Completeness", f"{quality['filled_core']}/{quality['total_core']}")
            st.progress(quality["core_ratio"])
        with q2:
            st.metric("Confidence", quality["confidence_label"])
            st.caption(f"Evidence coverage: {quality['filled_evidence']}/{quality['total_evidence']}")
            st.progress(quality["confidence_score"])

        st.markdown("**Title**")
        st.info(summary_data.get("title", "Not available"))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Dataset**")
            st.write(summary_data.get("dataset", "Not available"))
        with c2:
            st.markdown("**Metrics**")
            st.write(format_metrics(summary_data.get("metrics", [])))
        with c3:
            st.markdown("**Limitations**")
            st.write(summary_data.get("limitations", "Not available"))

        st.markdown("**Research Problem**")
        st.write(summary_data.get("research_problem", "Not available"))

        st.markdown("**Method**")
        st.write(summary_data.get("method", "Not available"))

        st.markdown("**Main Results**")
        st.write(summary_data.get("main_results", "Not available"))

        st.markdown("**Novelty**")
        st.write(summary_data.get("novelty", "Not available"))

        with st.expander("Evidence snippets"):
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Problem Evidence**")
                st.info(summary_data.get("evidence_problem", "Not clearly stated"))

                st.markdown("**Method Evidence**")
                st.info(summary_data.get("evidence_method", "Not clearly stated"))

            with e2:
                st.markdown("**Results Evidence**")
                st.info(summary_data.get("evidence_results", "Not clearly stated"))

                st.markdown("**Novelty Evidence**")
                st.info(summary_data.get("evidence_novelty", "Not clearly stated"))

        with st.expander("Download summary"):
            st.json(summary_data)

            st.download_button(
                "Download summary JSON",
                data=json.dumps(summary_data, ensure_ascii=False, indent=2),
                file_name=f"summary_{make_safe_name(file_name, 40)}.json",
                mime="application/json",
                key=f"download_summary_json_{file_name}",
            )

            md_text = summary_to_markdown(summary_data, paper_label, file_name)
            st.download_button(
                "Download summary Markdown",
                data=md_text,
                file_name=f"summary_{make_safe_name(file_name, 40)}.md",
                mime="text/markdown",
                key=f"download_summary_md_{file_name}",
            )

            st.download_button(
                "Download summary TXT",
                data=md_text,
                file_name=f"summary_{make_safe_name(file_name, 40)}.txt",
                mime="text/plain",
                key=f"download_summary_txt_{file_name}",
            )


def make_takeaways(comparison: dict) -> list[str]:
    takeaways = []

    for field in ["key_difference", "practical_takeaway", "method_gap", "evaluation_gap"]:
        value = comparison.get(field, "").strip()
        if value:
            first = value.split(". ")[0].strip()
            if first and not first.endswith("."):
                first += "."
            takeaways.append(first)

    return takeaways


def display_related_work_card(related_work: dict, compare_key: str, file1: str, file2: str):
    quality = compute_related_work_quality(related_work)

    with st.container(border=True):
        st.markdown("### Related Work Draft")

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Readiness", f"{quality['filled']}/{quality['total']}")
            st.progress(quality["ratio"])
        with r2:
            st.metric("Quality", quality["label"])

        st.markdown("**Paragraph**")
        st.text_area(
            label="Related Work Paragraph",
            value=related_work.get("related_work_paragraph", "-"),
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

        st.markdown("**Positioning Statement**")
        st.info(related_work.get("positioning_statement", "-"))

        with st.expander("Download related work"):
            st.json(related_work)

            st.download_button(
                "Download related work JSON",
                data=json.dumps(related_work, ensure_ascii=False, indent=2),
                file_name=f"related_work_{make_safe_name(compare_key, 50)}.json",
                mime="application/json",
                key=f"download_related_work_json_{compare_key}",
            )

            md_text = related_work_to_markdown(related_work, file1, file2)

            st.download_button(
                "Download related work Markdown",
                data=md_text,
                file_name=f"related_work_{make_safe_name(compare_key, 50)}.md",
                mime="text/markdown",
                key=f"download_related_work_md_{compare_key}",
            )

            st.download_button(
                "Download related work TXT",
                data=md_text,
                file_name=f"related_work_{make_safe_name(compare_key, 50)}.txt",
                mime="text/plain",
                key=f"download_related_work_txt_{compare_key}",
            )


def display_recommendation_card(recommendation: dict, compare_key: str, file1: str, file2: str):
    quality = compute_recommendation_quality(recommendation)

    with st.container(border=True):
        st.markdown("### Recommendation Panel")

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Recommendation Coverage", f"{quality['filled']}/{quality['total']}")
            st.progress(quality["ratio"])
        with r2:
            st.metric("Recommendation Quality", quality["label"])

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**More Practical Paper**")
            st.info(recommendation.get("more_practical_paper", "-"))

            st.markdown("**Better Baseline Paper**")
            st.info(recommendation.get("better_baseline_paper", "-"))

            st.markdown("**Better for Implementation Reference**")
            st.info(recommendation.get("better_for_implementation_reference", "-"))

        with c2:
            st.markdown("**More Novel Paper**")
            st.info(recommendation.get("more_novel_paper", "-"))

            st.markdown("**Better for Research Inspiration**")
            st.info(recommendation.get("better_for_research_inspiration", "-"))

        st.markdown("**Recommendation Reasoning**")
        st.write(recommendation.get("recommendation_reasoning", "-"))

        with st.expander("Download recommendation"):
            st.json(recommendation)

            st.download_button(
                "Download recommendation JSON",
                data=json.dumps(recommendation, ensure_ascii=False, indent=2),
                file_name=f"recommendation_{make_safe_name(compare_key, 50)}.json",
                mime="application/json",
                key=f"download_recommendation_json_{compare_key}",
            )

            md_text = recommendation_to_markdown(recommendation, file1, file2)

            st.download_button(
                "Download recommendation Markdown",
                data=md_text,
                file_name=f"recommendation_{make_safe_name(compare_key, 50)}.md",
                mime="text/markdown",
                key=f"download_recommendation_md_{compare_key}",
            )

            st.download_button(
                "Download recommendation TXT",
                data=md_text,
                file_name=f"recommendation_{make_safe_name(compare_key, 50)}.txt",
                mime="text/plain",
                key=f"download_recommendation_txt_{compare_key}",
            )


# =========================
# History Panel
# =========================
def list_recent_output_files(prefix: str, limit: int = 8) -> list[Path]:
    files = list(OUTPUT_DIR.glob(f"{prefix}_*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def load_json_file(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def history_label(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} — {path.name}"


def render_history_section(title: str, prefix: str, section_key: str):
    files = list_recent_output_files(prefix)

    with st.sidebar.expander(title, expanded=False):
        if not files:
            st.caption("Belum ada file tersimpan.")
            return

        options = {history_label(f): f for f in files}
        selected_label = st.selectbox(
            f"Pilih {title}",
            list(options.keys()),
            key=f"history_select_{section_key}",
        )

        selected_path = options[selected_label]
        data = load_json_file(selected_path)

        st.caption(selected_path.name)

        if data is None:
            st.error("Gagal membaca file JSON.")
            return

        st.download_button(
            f"Download {title}",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=selected_path.name,
            mime="application/json",
            key=f"history_download_{section_key}",
            use_container_width=True,
        )

        st.json(data)


def render_history_panel():
    st.sidebar.markdown("## Recent Outputs")

    render_history_section("Recent Summaries", "summary", "summary")
    render_history_section("Recent Comparisons", "comparison", "comparison")
    render_history_section("Recent Related Work", "related_work", "related_work")
    render_history_section("Recent Recommendations", "recommendation", "recommendation")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Current Session", use_container_width=True):
        st.session_state.basic_infos = {}
        st.session_state.paper_summaries = {}
        st.session_state.comparison_results = {}
        st.session_state.related_work_results = {}
        st.session_state.recommendation_results = {}
        st.rerun()


render_history_panel()


# =========================
# Main UI
# =========================
uploaded_files = st.file_uploader(
    "Upload maksimal 2 paper PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 2:
        st.warning("Hanya 2 file pertama yang akan diproses.")
        uploaded_files = uploaded_files[:2]

    docs = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        save_path = UPLOAD_DIR / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            paper_text = extract_text_from_pdf(str(save_path))
        except Exception as e:
            st.error(f"Gagal membaca PDF {uploaded_file.name}: {e}")
            continue

        if not paper_text.strip():
            st.warning(f"Teks PDF tidak berhasil diekstrak untuk {uploaded_file.name}.")
            continue

        docs.append(
            {
                "paper_id": f"paper_{idx}",
                "paper_label": f"Paper {idx}",
                "file_name": uploaded_file.name,
                "paper_text": paper_text,
            }
        )

    if not docs:
        st.stop()

    st.markdown("## Uploaded Papers")

    uploaded_rows = []
    for doc in docs:
        uploaded_rows.append(
            {
                "Paper": doc["paper_label"],
                "File Name": doc["file_name"],
                "Characters": f"{len(doc['paper_text']):,}",
            }
        )
    st.dataframe(pd.DataFrame(uploaded_rows), use_container_width=True, hide_index=True)

    st.markdown("## Preview")
    preview_tabs = st.tabs([doc["paper_label"] for doc in docs])

    for tab, doc in zip(preview_tabs, docs):
        with tab:
            st.caption(doc["file_name"])
            st.text_area(
                label=f"preview_{doc['file_name']}",
                value=clean_preview_text(doc["paper_text"]),
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )

    st.markdown("## Actions")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("Extract Basic Info", use_container_width=True):
            for doc in docs:
                info = extract_basic_info(doc["paper_text"], doc["file_name"])
                st.session_state.basic_infos[doc["file_name"]] = info
                save_json("basic_info", doc["file_name"], info)

    with c2:
        if st.button("Summarize with AI", use_container_width=True):
            if client is None:
                st.error("OPENAI_API_KEY belum ditemukan di file .env")
                st.stop()

            with st.spinner("Membuat summary untuk paper yang diupload..."):
                try:
                    for doc in docs:
                        summary = summarize_paper_with_ai(doc["paper_text"])
                        st.session_state.paper_summaries[doc["file_name"]] = summary
                        save_json("summary", doc["file_name"], summary)
                except Exception as e:
                    error_text = str(e)
                    if "insufficient_quota" in error_text:
                        st.error("Quota API belum tersedia. Aktifkan billing API terlebih dahulu.")
                    else:
                        st.error(f"Terjadi error saat summary: {e}")

    with c3:
        if st.button("Compare 2 Papers", use_container_width=True):
            if len(docs) != 2:
                st.warning("Upload tepat 2 paper untuk compare.")
            elif client is None:
                st.error("OPENAI_API_KEY belum ditemukan di file .env")
            else:
                try:
                    for doc in docs:
                        if doc["file_name"] not in st.session_state.paper_summaries:
                            summary = summarize_paper_with_ai(doc["paper_text"])
                            st.session_state.paper_summaries[doc["file_name"]] = summary
                            save_json("summary", doc["file_name"], summary)

                    file1 = docs[0]["file_name"]
                    file2 = docs[1]["file_name"]

                    summary1 = st.session_state.paper_summaries[file1]
                    summary2 = st.session_state.paper_summaries[file2]

                    comparison = compare_two_summaries_with_ai(summary1, summary2)
                    compare_key = f"{Path(file1).stem}__VS__{Path(file2).stem}"
                    st.session_state.comparison_results[compare_key] = comparison
                    save_json("comparison", compare_key, comparison)

                except Exception as e:
                    error_text = str(e)
                    if "insufficient_quota" in error_text:
                        st.error("Quota API belum tersedia. Aktifkan billing API terlebih dahulu.")
                    else:
                        st.error(f"Terjadi error saat comparison: {e}")

    with c4:
        if st.button("Generate Related Work", use_container_width=True):
            if len(docs) != 2:
                st.warning("Upload tepat 2 paper untuk membuat related work.")
            elif client is None:
                st.error("OPENAI_API_KEY belum ditemukan di file .env")
            else:
                try:
                    for doc in docs:
                        if doc["file_name"] not in st.session_state.paper_summaries:
                            summary = summarize_paper_with_ai(doc["paper_text"])
                            st.session_state.paper_summaries[doc["file_name"]] = summary
                            save_json("summary", doc["file_name"], summary)

                    file1 = docs[0]["file_name"]
                    file2 = docs[1]["file_name"]

                    summary1 = st.session_state.paper_summaries[file1]
                    summary2 = st.session_state.paper_summaries[file2]

                    compare_key = f"{Path(file1).stem}__VS__{Path(file2).stem}"

                    if compare_key not in st.session_state.comparison_results:
                        comparison = compare_two_summaries_with_ai(summary1, summary2)
                        st.session_state.comparison_results[compare_key] = comparison
                        save_json("comparison", compare_key, comparison)

                    comparison = st.session_state.comparison_results[compare_key]

                    related_work = generate_related_work_with_ai(summary1, summary2, comparison)
                    st.session_state.related_work_results[compare_key] = related_work
                    save_json("related_work", compare_key, related_work)

                except Exception as e:
                    error_text = str(e)
                    if "insufficient_quota" in error_text:
                        st.error("Quota API belum tersedia. Aktifkan billing API terlebih dahulu.")
                    else:
                        st.error(f"Terjadi error saat membuat related work: {e}")

    with c5:
        if st.button("Generate Recommendation", use_container_width=True):
            if len(docs) != 2:
                st.warning("Upload tepat 2 paper untuk membuat recommendation panel.")
            elif client is None:
                st.error("OPENAI_API_KEY belum ditemukan di file .env")
            else:
                try:
                    for doc in docs:
                        if doc["file_name"] not in st.session_state.paper_summaries:
                            summary = summarize_paper_with_ai(doc["paper_text"])
                            st.session_state.paper_summaries[doc["file_name"]] = summary
                            save_json("summary", doc["file_name"], summary)

                    file1 = docs[0]["file_name"]
                    file2 = docs[1]["file_name"]

                    summary1 = st.session_state.paper_summaries[file1]
                    summary2 = st.session_state.paper_summaries[file2]

                    compare_key = f"{Path(file1).stem}__VS__{Path(file2).stem}"

                    if compare_key not in st.session_state.comparison_results:
                        comparison = compare_two_summaries_with_ai(summary1, summary2)
                        st.session_state.comparison_results[compare_key] = comparison
                        save_json("comparison", compare_key, comparison)

                    comparison = st.session_state.comparison_results[compare_key]

                    recommendation = generate_recommendation_with_ai(summary1, summary2, comparison)
                    st.session_state.recommendation_results[compare_key] = recommendation
                    save_json("recommendation", compare_key, recommendation)

                except Exception as e:
                    error_text = str(e)
                    if "insufficient_quota" in error_text:
                        st.error("Quota API belum tersedia. Aktifkan billing API terlebih dahulu.")
                    else:
                        st.error(f"Terjadi error saat membuat recommendation panel: {e}")

    current_file_names = [doc["file_name"] for doc in docs]
    paper_label_map = {doc["file_name"]: doc["paper_label"] for doc in docs}

    available_basic_infos = [f for f in current_file_names if f in st.session_state.basic_infos]
    if available_basic_infos:
        st.markdown("## Basic Info Results")
        cols = st.columns(len(available_basic_infos))
        for col, file_name in zip(cols, available_basic_infos):
            with col:
                display_basic_info_card(
                    st.session_state.basic_infos[file_name],
                    paper_label_map[file_name],
                )

    available_summaries = [f for f in current_file_names if f in st.session_state.paper_summaries]
    if available_summaries:
        st.markdown("## AI Summary Results")
        cols = st.columns(len(available_summaries))
        for col, file_name in zip(cols, available_summaries):
            with col:
                display_summary_card(
                    st.session_state.paper_summaries[file_name],
                    paper_label_map[file_name],
                    file_name,
                )

    if len(docs) == 2:
        compare_key = f"{Path(docs[0]['file_name']).stem}__VS__{Path(docs[1]['file_name']).stem}"

        if compare_key in st.session_state.comparison_results:
            st.markdown("## Comparison Result")

            summary1 = st.session_state.paper_summaries[docs[0]["file_name"]]
            summary2 = st.session_state.paper_summaries[docs[1]["file_name"]]
            comparison = st.session_state.comparison_results[compare_key]
            comparison_quality = compute_comparison_quality(comparison)

            with st.container(border=True):
                st.markdown("### Paper Mapping")
                st.write(f"**Paper 1:** {docs[0]['file_name']}")
                st.write(f"**Paper 2:** {docs[1]['file_name']}")

            cq1, cq2 = st.columns(2)
            with cq1:
                st.metric("Comparison Coverage", f"{comparison_quality['filled']}/{comparison_quality['total']}")
                st.progress(comparison_quality["ratio"])
            with cq2:
                st.metric("Comparison Quality", comparison_quality["label"])

            comparison_df = build_comparison_dataframe(summary1, summary2)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            takeaways = make_takeaways(comparison)
            if takeaways:
                with st.container(border=True):
                    st.markdown("### Key Takeaways")
                    for item in takeaways:
                        st.markdown(f"- {item}")

            i1, i2 = st.columns(2)
            with i1:
                with st.container(border=True):
                    st.markdown("### Key Difference")
                    st.write(comparison.get("key_difference", "-"))

                with st.container(border=True):
                    st.markdown("### Paper 1 Strength")
                    st.write(comparison.get("paper_1_strength", "-"))

            with i2:
                with st.container(border=True):
                    st.markdown("### Paper 2 Strength")
                    st.write(comparison.get("paper_2_strength", "-"))

                with st.container(border=True):
                    st.markdown("### Practical Takeaway")
                    st.write(comparison.get("practical_takeaway", "-"))

            st.markdown("### Structured Research Gap")
            g1, g2 = st.columns(2)
            with g1:
                with st.container(border=True):
                    st.markdown("**Method Gap**")
                    st.write(comparison.get("method_gap", "-"))
                with st.container(border=True):
                    st.markdown("**Dataset Gap**")
                    st.write(comparison.get("dataset_gap", "-"))
            with g2:
                with st.container(border=True):
                    st.markdown("**Evaluation Gap**")
                    st.write(comparison.get("evaluation_gap", "-"))
                with st.container(border=True):
                    st.markdown("**Implementation Gap**")
                    st.write(comparison.get("implementation_gap", "-"))

            with st.container(border=True):
                st.markdown("### Future Direction")
                st.write(comparison.get("future_direction", "-"))

            with st.expander("Download comparison"):
                st.json(comparison)

                st.download_button(
                    "Download comparison JSON",
                    data=json.dumps(comparison, ensure_ascii=False, indent=2),
                    file_name=f"comparison_{make_safe_name(compare_key, 50)}.json",
                    mime="application/json",
                    key="download_comparison_json",
                )

                md_text = comparison_to_markdown(
                    comparison,
                    docs[0]["file_name"],
                    docs[1]["file_name"],
                )

                st.download_button(
                    "Download comparison Markdown",
                    data=md_text,
                    file_name=f"comparison_{make_safe_name(compare_key, 50)}.md",
                    mime="text/markdown",
                    key="download_comparison_md",
                )

                st.download_button(
                    "Download comparison TXT",
                    data=md_text,
                    file_name=f"comparison_{make_safe_name(compare_key, 50)}.txt",
                    mime="text/plain",
                    key="download_comparison_txt",
                )

            if compare_key in st.session_state.related_work_results:
                st.markdown("## Related Work Output")
                display_related_work_card(
                    st.session_state.related_work_results[compare_key],
                    compare_key,
                    docs[0]["file_name"],
                    docs[1]["file_name"],
                )

            if compare_key in st.session_state.recommendation_results:
                st.markdown("## Recommendation Output")
                display_recommendation_card(
                    st.session_state.recommendation_results[compare_key],
                    compare_key,
                    docs[0]["file_name"],
                    docs[1]["file_name"],
                )
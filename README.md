# Research Assistant Agent for Scientific Literature Review

Research Assistant Agent adalah aplikasi berbasis Streamlit yang dirancang untuk membantu membaca, merangkum, membandingkan, dan menyusun insight dari paper ilmiah secara lebih terstruktur. Aplikasi ini mendukung alur kerja literature review dengan memanfaatkan ekstraksi teks PDF, ringkasan berbasis AI, perbandingan dua paper, related work draft, serta recommendation panel.

---

## Features

Aplikasi ini memiliki beberapa fitur utama:

- Upload maksimal 2 paper PDF
- Preview hasil ekstraksi teks PDF
- Extract Basic Info:
  - title guess
  - authors guess
  - abstract preview
- AI Summary untuk setiap paper:
  - title
  - research problem
  - method
  - dataset
  - metrics
  - main results
  - novelty
  - limitations
- Evidence snippets untuk mendukung hasil summary
- Compare 2 Papers
- Structured Research Gap:
  - method gap
  - dataset gap
  - evaluation gap
  - implementation gap
- Related Work Draft
- Recommendation Panel:
  - more practical paper
  - more novel paper
  - better baseline paper
  - better for implementation reference
  - better for research inspiration
- Confidence / Completeness indicators
- Download output dalam format:
  - JSON
  - Markdown
  - TXT
- Recent Outputs panel di sidebar

---

## Tech Stack

- Python
- Streamlit
- OpenAI API
- PyPDF
- Pandas
- python-dotenv

---

## Project Structure

```text
research-assistant-agent/
├─ app.py
├─ prompts.py
├─ extractor.py
├─ requirements.txt
├─ .env
├─ .gitignore
├─ README.md
├─ data/
│  └─ uploads/
├─ outputs/
├─ screenshots/
└─ examples/
import re
import heapq
import html
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# -------------------- Page setup --------------------

st.set_page_config(
    page_title="Verba | Smart Text Summarizer",
    page_icon="📚",
    layout="wide"
)

st.session_state.setdefault("dark_mode", True)
st.session_state.setdefault("input_text", "")
st.session_state.setdefault("summary_result", None)

with st.sidebar:
    st.markdown("## ⚙️ Preferences")
    st.toggle("🌙 Dark Mode", key="dark_mode")
    st.caption("SDG 4 · Quality Education")

if st.session_state.dark_mode:
    bg = "#171923"
    panel = "#222533"
    text = "#F3F4F8"
    muted = "#A9B0C3"
    accent = "#8B7CFF"
else:
    bg = "#F4F6FB"
    panel = "#FFFFFF"
    text = "#202338"
    muted = "#626B80"
    accent = "#6554D9"


# -------------------- Styling --------------------

st.markdown(f"""
<style>
    .stApp {{ background: {bg}; color: {text}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{
        padding-top: 1.6rem;
        max-width: 1450px;
    }}
    h1, h2, h3, p, label, .stMarkdown {{
        color: {text};
    }}
    .hero {{
        background: linear-gradient(120deg, {panel}, {bg});
        border: 1px solid {accent}66;
        border-radius: 22px;
        padding: 25px 28px;
        margin-bottom: 22px;
    }}
    .eyebrow {{
        color: {accent};
        font-weight: 750;
        letter-spacing: .12em;
        font-size: .78rem;
    }}
    .hero h1 {{
        margin: 5px 0 7px 0;
        font-size: 2.25rem;
    }}
    .hero p {{ color: {muted}; }}
    .panel {{
        background: {panel};
        border: 1px solid #88888822;
        border-radius: 18px;
        padding: 18px;
    }}
    div[data-testid="stTextArea"] textarea {{
        background: {panel};
        color: {text};
        border-radius: 12px;
    }}
    div[data-testid="stFileUploader"] {{
        background: {panel};
        border-radius: 12px;
        padding: 8px;
    }}
    div.stButton > button,
    div.stDownloadButton > button {{
        border-radius: 11px;
        font-weight: 700;
        min-height: 2.8rem;
    }}
    .metric-card {{
        background: {panel};
        border: 1px solid #88888822;
        border-radius: 15px;
        padding: 15px;
        min-height: 95px;
    }}
    .metric-label {{
        color: {muted};
        font-size: .82rem;
    }}
    .metric-value {{
        color: {text};
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 4px;
    }}
    .sdg-badge {{
        display: inline-block;
        background: #C5192D;
        color: white;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: .8rem;
        font-weight: 800;
    }}
    .small-note {{
        color: {muted};
        font-size: .85rem;
    }}
</style>
""", unsafe_allow_html=True)


# -------------------- Application heading --------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">LEARN SMARTER · STUDY BETTER</div>
    <h1>📚 Verba</h1>
    <p>
        Smart Text Summarizer · Turn lengthy study material
        into clear, useful key points.
    </p>
    <br>
    <span class="sdg-badge">SDG 4 · QUALITY EDUCATION</span>
</div>
""", unsafe_allow_html=True)


# -------------------- Text processing --------------------

STOP_WORDS = set("""
a an the and or but if while is am are was were be been being
have has had do does did of to in on for with as by at from
into about than then this that these those it its he she they
them we you your i my our their his her not no so too very can
could should would will just also such which who whom what when
where why how all any each few more most other some only own same
s t d ll ve re
""".split())


def split_sentences(text):
    """Split text into sentences while retaining original order."""
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return []

    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"“\'])',
        cleaned
    )

    return [part.strip() for part in parts if part.strip()]


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(
        r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?",
        text.lower()
    )


# -------------------- Dynamic Programming: LCS --------------------

def lcs_length(a, b):
    """Calculate Longest Common Subsequence using dynamic programming."""
    previous = [0] * (len(b) + 1)

    for word_a in a:
        current = [0]

        for j, word_b in enumerate(b, start=1):
            if word_a == word_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))

        previous = current

    return previous[-1]


def similarity_lcs(sentence_a, sentence_b):
    """Calculate normalized LCS similarity between two sentences."""
    a = tokenize(sentence_a)
    b = tokenize(sentence_b)

    if not a or not b:
        return 0.0

    return lcs_length(a, b) / max(len(a), len(b))


# -------------------- Summarization algorithms --------------------

def summarize_text(text, target_count, redundancy_threshold=0.72):
    sentences = split_sentences(text)
    all_tokens = tokenize(text)

    if not sentences:
        return None

    # Algorithm 1: Hash Map / word-frequency counting.
    useful_tokens = [
        word for word in all_tokens
        if word not in STOP_WORDS and len(word) > 1
    ]

    frequencies = Counter(useful_tokens)

    if not frequencies:
        frequencies = Counter(all_tokens)

    # Calculate each sentence's importance score.
    scored = []

    for index, sentence in enumerate(sentences):
        words = [
            word for word in tokenize(sentence)
            if word not in STOP_WORDS and len(word) > 1
        ]

        if not words:
            words = tokenize(sentence)

        score = (
            sum(frequencies.get(word, 0) for word in words)
            / max(len(words), 1)
        )

        scored.append((score, index, sentence))

    # Algorithm 2: Priority Queue using heapq.
    # Negative scores make the highest score come out first.
    heap = [
        (-score, index, sentence)
        for score, index, sentence in scored
    ]

    heapq.heapify(heap)

    ranked = []

    while heap:
        negative_score, index, sentence = heapq.heappop(heap)

        ranked.append({
            "score": -negative_score,
            "index": index,
            "sentence": sentence
        })

    # Algorithm 3: Dynamic Programming / LCS.
    # Avoid selecting sentences that are too similar.
    selected = []
    rejected_as_redundant = 0

    target_count = max(1, min(target_count, len(sentences)))

    for item in ranked:
        if len(selected) >= target_count:
            break

        duplicate = any(
            similarity_lcs(
                item["sentence"],
                existing["sentence"]
            ) >= redundancy_threshold
            for existing in selected
        )

        if duplicate:
            rejected_as_redundant += 1
            continue

        selected.append(item)

    # Fill remaining slots if redundancy filtering was too strict.
    if len(selected) < target_count:
        selected_indices = {
            item["index"] for item in selected
        }

        for item in ranked:
            if len(selected) >= target_count:
                break

            if item["index"] not in selected_indices:
                selected.append(item)
                selected_indices.add(item["index"])

    # Display selected sentences in original document order.
    selected.sort(key=lambda item: item["index"])

    summary_sentences = [
        item["sentence"] for item in selected
    ]

    original_words = len(all_tokens)

    summary_words = sum(
        len(tokenize(sentence))
        for sentence in summary_sentences
    )

    reduction = (
        (1 - summary_words / original_words) * 100
        if original_words else 0
    )

    return {
        "summary": "\n".join(
            f"• {sentence}" for sentence in summary_sentences
        ),
        "summary_sentences": summary_sentences,
        "sentences": sentences,
        "frequencies": frequencies,
        "ranked": ranked,
        "original_words": original_words,
        "summary_words": summary_words,
        "reduction": max(0.0, reduction),
        "vocabulary": len(set(all_tokens)),
        "rejected": rejected_as_redundant,
    }


# -------------------- File reading: TXT, DOCX and PDF --------------------

def read_uploaded_file(uploaded_file):
    """Extract text from TXT, DOCX or searchable PDF files."""
    name = uploaded_file.name.lower()

    # TXT files
    if name.endswith(".txt"):
        raw = uploaded_file.getvalue()

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    # Word documents
    elif name.endswith(".docx"):
        if Document is None:
            raise RuntimeError(
                "Install python-docx: "
                "python -m pip install python-docx"
            )

        uploaded_file.seek(0)
        document = Document(uploaded_file)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    # PDF documents
    elif name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "Install pypdf: "
                "python -m pip install pypdf"
            )

        uploaded_file.seek(0)

        try:
            reader = PdfReader(uploaded_file, strict=False)

            if reader.is_encrypted:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Please upload an unlocked PDF."
                )

            pages_text = []

            for page in reader.pages:
                page_text = page.extract_text() or ""

                if page_text.strip():
                    pages_text.append(page_text.strip())

            extracted_text = "\n\n".join(pages_text)

            if not extracted_text.strip():
                raise ValueError(
                    "No selectable text was found in this PDF. "
                    "It may be scanned or image-only. "
                    "Run OCR on it to create a searchable PDF."
                )

            return extracted_text

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                f"Could not read this PDF: {exc}"
            ) from exc

    else:
        raise ValueError(
            "Unsupported file. Upload TXT, DOCX or PDF."
        )


# -------------------- Main workspace --------------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📝 Input document")

    st.caption(
        "Paste your study material or upload a TXT, DOCX or PDF file."
    )

    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "docx", "pdf"],
        help=(
            "Supported formats: TXT, DOCX and PDF. "
            "Configured maximum: 1 GB per file, subject to "
            "hosting limits and available memory."
        )
    )

    if uploaded is not None:
        try:
            uploaded_text = read_uploaded_file(uploaded)

            st.session_state.input_text = uploaded_text
            st.success(f"Loaded: {uploaded.name}")

        except Exception as exc:
            st.error(str(exc))

    input_text = st.text_area(
        "Document text",
        key="input_text",
        height=310,
        placeholder=(
            "Paste a long article, chapter, "
            "or lecture notes here..."
        ),
        label_visibility="collapsed"
    )

    total_sentences = len(split_sentences(input_text))

    st.markdown(
        f"<div class='small-note'>"
        f"📄 {len(tokenize(input_text)):,} words · "
        f"{total_sentences:,} sentences detected"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### 🎚️ Summary settings")

    length_mode = st.radio(
        "Choose summary length by",
        ["Number of sentences", "Text reduction (%)"],
        horizontal=True
    )

    if length_mode == "Number of sentences":
        target_count = st.slider(
            "Sentences in summary",
            min_value=1,
            max_value=max(1, min(30, total_sentences or 10)),
            value=min(5, max(1, total_sentences or 5))
        )

    else:
        reduction_target = st.slider(
            "Target text reduction",
            min_value=10,
            max_value=90,
            value=50,
            step=5
        )

        target_count = max(
            1,
            round(
                (total_sentences or 1)
                * (100 - reduction_target) / 100
            )
        )

        st.caption(
            f"Approximately {target_count} sentence(s) "
            "will be selected."
        )

    generate = st.button(
        "✨ Generate Summary",
        type="primary",
        use_container_width=True
    )

    if generate:
        if not input_text.strip():
            st.warning(
                "Please paste text or upload a document "
                "before generating a summary."
            )
        else:
            with st.spinner(
                "Analyzing word frequencies, ranking sentences, "
                "and checking redundancy..."
            ):
                try:
                    st.session_state.summary_result = summarize_text(
                        input_text,
                        target_count
                    )
                except Exception as exc:
                    st.error(f"Could not summarize the document: {exc}")


# -------------------- Summary output --------------------

with right:
    st.markdown("### ✨ Summary output")

    result = st.session_state.summary_result

    if result:
        st.caption(
            "Key sentences selected by the summarization algorithms"
        )

        st.markdown(
            "<div class='panel'>",
            unsafe_allow_html=True
        )

        for sentence in result["summary_sentences"]:
            st.markdown(f"- {sentence}")

        st.markdown("</div>", unsafe_allow_html=True)

        summary_plain = "\n".join(
            result["summary_sentences"]
        )

        # Browser clipboard button
        components.html(
            f"""
            <div style="font-family:sans-serif;">
                <button id="copyBtn"
                    style="background:{accent};color:white;
                    border:0;border-radius:9px;padding:10px 15px;
                    font-weight:700;cursor:pointer;">
                    📋 Copy to Clipboard
                </button>
                <span id="copyStatus"
                    style="margin-left:10px;font-size:13px;
                    color:{muted};"></span>
            </div>

            <script>
                const textToCopy = {html.escape(summary_plain).__repr__()};

                document.getElementById('copyBtn')
                    .addEventListener('click', async () => {{
                        const status =
                            document.getElementById('copyStatus');

                        try {{
                            await navigator.clipboard.writeText(
                                textToCopy
                            );
                            status.textContent = 'Copied!';
                        }} catch (e) {{
                            status.textContent =
                                'Clipboard blocked. Select and copy '
                                + 'the summary manually.';
                        }}
                    }});
            </script>
            """,
            height=55
        )

        st.download_button(
            "⬇️ Download summary (.txt)",
            data=summary_plain,
            file_name="verba_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:
        st.markdown(
            f"""
            <div class="panel"
                style="min-height:310px;display:flex;
                align-items:center;justify-content:center;
                text-align:center;">
                <div>
                    <div style="font-size:2.8rem;">🪄</div>
                    <h3>Your summary will appear here</h3>
                    <p style="color:{muted};">
                        Add a document on the left and click
                        <b>Generate Summary</b> to get started.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------- Key Metrics dashboard --------------------

st.markdown("---")
st.markdown("## 📊 Key Metrics")

st.caption(
    "A quick view of what the algorithms found in your document."
)

result = st.session_state.summary_result

if result:
    metrics = [
        ("Original words", f"{result['original_words']:,}"),
        ("Summary words", f"{result['summary_words']:,}"),
        ("Text reduction", f"{result['reduction']:.1f}%"),
        ("Unique vocabulary", f"{result['vocabulary']:,}"),
        ("Sentences analyzed", f"{len(result['sentences']):,}"),
        ("Redundancies filtered", f"{result['rejected']:,}"),
    ]

    cols = st.columns(3)

    for i, (label, value) in enumerate(metrics):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 🧠 Algorithm activity")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**1. Hash Map**")

        st.write(
            f"Counted **{len(result['frequencies']):,}** "
            "useful unique word(s)."
        )

        common = result["frequencies"].most_common(5)

        if common:
            st.caption(
                "Top words: "
                + ", ".join(
                    f"{word} ({count})"
                    for word, count in common
                )
            )

    with c2:
        st.markdown("**2. Priority Queue (Heap)**")

        st.write(
            f"Ranked **{len(result['ranked']):,}** "
            "sentences by word-frequency score."
        )

        if result["ranked"]:
            highest = result["ranked"][0]["sentence"]

            st.caption(
                "Highest-ranked sentence: "
                + highest[:180]
                + ("…" if len(highest) > 180 else "")
            )

    with c3:
        st.markdown("**3. Dynamic Programming (LCS)**")

        st.write(
            "Compared sentence token sequences using "
            "LCS similarity to identify overlapping content."
        )

        st.caption(
            "Redundant candidates skipped before fill-back: "
            f"{result['rejected']}"
        )

else:
    st.info(
        "Generate a summary to see document metrics "
        "and algorithm activity here."
    )


# -------------------- Explanation --------------------

with st.expander("ℹ️ How the summarizer works"):
    st.markdown("""
    1. **Hash Map (`Counter`):** counts the frequency of useful words.
    2. **Priority Queue (`heapq`):** ranks sentences by word frequency.
    3. **Dynamic Programming (LCS):** compares word sequences to
       identify highly overlapping sentences.
    4. **Original order:** selected sentences are returned in
       their original document order.

    This is an extractive summarizer: it selects sentences from
    the source rather than generating entirely new sentences.

    LCS is a sequence-similarity heuristic, not a full measure
    of semantic meaning.
    """)


# -------------------- Footer --------------------

st.markdown(
    f"""
    <p style="text-align:center;color:{muted};
    font-size:.8rem;margin-top:28px;">
        Verba · Built with Python + Streamlit ·
        SDG 4: Quality Education
    </p>
    """,
    unsafe_allow_html=True
)



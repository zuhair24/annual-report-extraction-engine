import io
import re

import fitz
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Annual Report Extraction Engine",
    layout="wide"
)

st.title("Annual Report Extraction & Reconciliation Engine")
st.caption(
    "Extract financial figures from annual report PDFs, "
    "review page evidence, and download results."
)

METRICS = [
    "Revenue",
    "Profit After Tax",
    "Total Assets",
    "Total Liabilities",
    "Total Borrowings",
    "Cash and Cash Equivalents",
]

PATTERNS = {
    "Revenue": r"\b(?:revenue from operations|total revenue|revenue)\b",
    "Profit After Tax": r"\b(?:profit after tax|profit for the year|net profit after tax|net profit)\b",
    "Total Assets": r"\btotal assets\b",
    "Total Liabilities": r"\btotal liabilities\b",
    "Total Borrowings": r"\b(?:total borrowings|total borrowings and lease liabilities)\b",
    "Cash and Cash Equivalents": r"\bcash and cash equivalents\b",
}


def get_page_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [
            (i + 1, page.get_text("text"))
            for i, page in enumerate(doc)
        ]
    finally:
        doc.close()


def _clean_line(text):
    """Normalize spacing while keeping the financial row readable."""
    return re.sub(r"\s+", " ", text).strip()


def _infer_unit(context):
    """Return a unit only when the PDF text gives a clear indication."""
    checks = [
        (r"₹\s*(?:in\s*)?crores?|INR\s*(?:in\s*)?crores?", "INR crore"),
        (r"₹\s*(?:in\s*)?lakhs?|INR\s*(?:in\s*)?lakhs?", "INR lakh"),
        (r"₹\s*(?:in\s*)?millions?|INR\s*(?:in\s*)?millions?", "INR million"),
        (r"₹\s*(?:in\s*)?billions?|INR\s*(?:in\s*)?billions?", "INR billion"),
        (r"\bUSD\s*(?:in\s*)?millions?\b|\$\s*(?:in\s*)?millions?", "USD million"),
        (r"\bUSD\b|\$", "USD"),
        (r"\bINR\b|\bRs\.?\b|₹", "INR"),
    ]
    for pattern, unit in checks:
        if re.search(pattern, context, re.IGNORECASE):
            return unit
    return "Unit not identified"


def _infer_period(lines, line_index):
    """Use a nearby explicit financial-year/date header; don't guess if absent."""
    start = max(0, line_index - 12)
    context = " ".join(lines[start:line_index + 1])
    patterns = [
        r"\bFY\s*20\d{2}(?:\s*[-–/]\s*(?:20)?\d{2})?\b",
        r"\b(?:year ended|year ending|as at|as of|for the year ended)\s*"
        r"(?:31\s+March|March\s+31|31\s+December|December\s+31)?[,]?\s*(20\d{2})\b",
        r"\b(20\d{2}\s*[-–/]\s*(?:20)?\d{2})\b",
        r"\b(31\s+March\s+20\d{2}|March\s+31,?\s+20\d{2})\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, context, re.IGNORECASE))
        if matches:
            return _clean_line(matches[-1].group(0))
    return "Not identified"


def _extract_amounts(text):
    """Find numeric amounts, excluding common note references and dates."""
    text = re.sub(
        r"\(?\s*(?:refer to )?notes?\s*\d+[a-z]?\s*\)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:31\s+March|March\s+31|31\s+December|December\s+31)"
        r"[,]?\s*20\d{2}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    number_pattern = re.compile(
        r"(?<![A-Za-z0-9/])"
        r"(?:₹\s*|Rs\.?\s*|INR\s*|USD\s*|\$\s*)?"
        r"\(?\s*-?\s*\d[\d,]*(?:\.\d+)?\s*\)?"
        r"(?![A-Za-z0-9/])",
        re.IGNORECASE,
    )
    amounts = []
    for match in number_pattern.finditer(text):
        raw = match.group(0).strip()
        # Ignore standalone years and tiny note/index numbers.
        bare = re.sub(r"^(?:₹|Rs\.?|INR|USD|\$)\s*", "", raw, flags=re.IGNORECASE)
        bare = bare.strip().strip("()").replace(",", "").replace(" ", "")
        if not bare or not re.fullmatch(r"-?\d+(?:\.\d+)?", bare):
            continue
        numeric = float(bare)
        if 1900 <= numeric <= 2099 and "." not in bare:
            continue
        if abs(numeric) < 1 and "." not in bare:
            continue
        amounts.append((raw, bare))
    return amounts


def extract_with_ai(client, company, pages):
    # Local, line-based extraction; no API key is needed.
    # It prefers amounts on the same financial-statement row as the label,
    # rather than the first number anywhere in a large text window.
    results = []
    seen = set()

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        lines = [_clean_line(line) for line in page_text.splitlines()]
        lines = [line for line in lines if line]

        for line_index, line in enumerate(lines):
            for metric, pattern in PATTERNS.items():
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    # Only inspect the text after the matched label on this row.
                    suffix = line[match.end():]
                    amounts = _extract_amounts(suffix)
                    evidence_line = line

                    # Some PDFs place the label and numeric columns on adjacent
                    # lines. Only use short, number-led following lines as fallback.
                    if not amounts:
                        following = []
                        for next_line in lines[line_index + 1:line_index + 4]:
                            if re.search(r"[A-Za-z]{3,}", next_line):
                                break
                            following.append(next_line)
                        if following:
                            candidate_text = " ".join(following)
                            amounts = _extract_amounts(candidate_text)
                            if amounts:
                                evidence_line = line + " | " + " | ".join(following)

                    if not amounts:
                        continue

                    raw_value, value = amounts[0]
                    # Avoid duplicating the same metric/amount/page candidate.
                    key = (page_num, metric, value, evidence_line)
                    if key in seen:
                        continue
                    seen.add(key)

                    context_start = max(0, line_index - 12)
                    context = " ".join(lines[context_start:min(len(lines), line_index + 3)])
                    results.append({
                        "company": company,
                        "metric": metric,
                        "reported_label": match.group(0),
                        "value": value,
                        "unit": _infer_unit(context),
                        "period": _infer_period(lines, line_index),
                        "pdf_page": page_num,
                        "evidence": evidence_line[:500],
                        "confidence": "Review required",
                    })

    return results


def reconciliation(df):
    columns = ["Company", "Period", "Check", "Result", "Details"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    checks = []

    for (company, period), group in df.groupby(
        ["company", "period"], dropna=False
    ):
        for metric, count in group["metric"].value_counts().items():
            if count > 1:
                checks.append({
                    "Company": company,
                    "Period": period,
                    "Check": f"Duplicate metric: {metric}",
                    "Result": "Flagged",
                    "Details": (
                        f"{count} candidate values found. "
                        "Check the correct statement and reporting period."
                    ),
                })

        checks.append({
            "Company": company,
            "Period": period,
            "Check": "Financial figures",
            "Result": "Manual review",
            "Details": (
                "Verify values, units, reporting periods, and page evidence "
                "against the original annual report."
            ),
        })

    return pd.DataFrame(checks, columns=columns)


with st.sidebar:
    st.header("Setup")
    st.write("No API key required.")
    st.warning(
        "Prototype only. Verify extracted figures against the original PDF."
    )

company = st.text_input(
    "Company name",
    placeholder="e.g., Example Industries Ltd."
)

uploaded = st.file_uploader(
    "Upload an annual report PDF",
    type=["pdf"]
)

if uploaded is not None and company.strip():
    pdf_bytes = uploaded.getvalue()
    pages = get_page_text(pdf_bytes)

    st.info(
        f"PDF loaded: {len(pages)} pages. "
        "Scanned PDFs may require OCR."
    )

    if st.button("Extract financial figures", type="primary"):
        with st.spinner("Reading pages and extracting figures..."):
            items = extract_with_ai(None, company.strip(), pages)
            df = pd.DataFrame(items)

            if df.empty:
                st.warning(
                    "No figures were extracted. Try a text-based PDF."
                )
            else:
                st.subheader("Extracted figures")
                st.dataframe(df, use_container_width=True)

                st.subheader("Reconciliation / review flags")
                checks = reconciliation(df)
                st.dataframe(checks, use_container_width=True)

                output = io.BytesIO()

                with pd.ExcelWriter(
                    output, engine="openpyxl"
                ) as writer:
                    df.to_excel(
                        writer,
                        sheet_name="Extracted Data",
                        index=False
                    )
                    checks.to_excel(
                        writer,
                        sheet_name="Review Checks",
                        index=False
                    )
                    pd.DataFrame({
                        "Standard Metrics": METRICS
                    }).to_excel(
                        writer,
                        sheet_name="Standard Metrics",
                        index=False
                    )

                st.download_button(
                    "Download Excel output",
                    data=output.getvalue(),
                    file_name=(
                        f"{company.strip().replace(' ', '_')}"
                        "_extraction.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

elif not company.strip() or uploaded is None:
    st.markdown("""
    **How to use**
    1. Enter a company name.
    2. Upload an annual report PDF.
    3. Click **Extract financial figures**.
    4. Review the extracted values and page evidence.
    5. Download the Excel workbook.
    """)

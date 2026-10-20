
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
    "Revenue": r"(?:revenue from operations|total revenue|revenue)",
    "Profit After Tax": r"(?:profit after tax|profit for the year|net profit)",
    "Total Assets": r"total assets",
    "Total Liabilities": r"total liabilities",
    "Total Borrowings": r"total borrowings",
    "Cash and Cash Equivalents": r"cash and cash equivalents",
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


def extract_with_ai(client, company, pages):
    # Local extraction; no API key is needed.
    results = []

    number_pattern = re.compile(
        r"(?:₹|Rs\.?|INR|\$)?\s*"
        r"\(?-?\d[\d,]*(?:\.\d+)?\)?",
        re.IGNORECASE
    )

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        for metric, pattern in PATTERNS.items():
            for match in re.finditer(pattern, page_text, re.IGNORECASE):
                nearby_text = page_text[
                    match.end():match.end() + 180
                ]

                number_match = number_pattern.search(nearby_text)

                if not number_match:
                    continue

                raw_value = number_match.group(0).strip()
                value = re.sub(
                    r"^(?:₹|Rs\.?|INR|\$)\s*",
                    "",
                    raw_value,
                    flags=re.IGNORECASE
                ).replace(",", "")

                if value.startswith("(") and value.endswith(")"):
                    value = "-" + value[1:-1]

                evidence_end = match.end() + number_match.end()

                results.append({
                    "company": company,
                    "metric": metric,
                    "reported_label": match.group(0),
                    "value": value,
                    "unit": "As reported in PDF",
                    "period": "",
                    "pdf_page": page_num,
                    "evidence": page_text[
                        max(0, match.start() - 40):
                        min(len(page_text), evidence_end + 40)
                    ].strip(),
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


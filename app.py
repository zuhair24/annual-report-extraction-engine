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
    # Prefer specific statement labels. Broad "revenue" matches are too noisy
    # in narrative text and note disclosures.
    "Revenue": r"\b(?:revenue from operations|revenue from contracts with customers|total revenue)\b",
    "Profit After Tax": r"\b(?:profit after tax|profit for the year attributable to owners|net profit after tax)\b",
    "Total Assets": r"^total assets$",
    "Total Liabilities": r"^total liabilities$",
    "Total Borrowings": r"\btotal borrowings\b",
    "Cash and Cash Equivalents": r"^cash and cash equivalents$",
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
    """Detect common financial-statement units from nearby or page-level text."""
    checks = [
        (r"\(\s*(?:₹|C)\s*in\s*crores?\s*\)|\bINR\s*(?:in\s*)?crores?\b|₹\s*in\s*crores?", "INR crore"),
        (r"\(\s*(?:₹|C)\s*in\s*lakhs?\s*\)|\bINR\s*(?:in\s*)?lakhs?\b|₹\s*in\s*lakhs?", "INR lakh"),
        (r"\(\s*(?:₹|C)\s*in\s*millions?\s*\)|\bINR\s*(?:in\s*)?millions?\b|₹\s*in\s*millions?", "INR million"),
        (r"\bUSD\s*(?:in\s*)?millions?\b|\$\s*in\s*millions?", "USD million"),
        (r"\bUSD\b|\$", "USD"),
        (r"\bINR\b|\bRs\.?\b|₹", "INR"),
    ]
    for pattern, unit in checks:
        if re.search(pattern, context, re.IGNORECASE):
            return unit
    return "Unit not identified"


def _infer_period(lines, line_index=None):
    """Prefer explicit statement date headers near the metric row."""
    context = " ".join(lines if line_index is None else lines[max(0, line_index - 8):line_index + 1])
    patterns = [
        r"\bFY\s*20\d{2}\s*[-–/]\s*(?:20)?\d{2}\b",
        r"\b(?:year ended|as at|as of|for the year ended)\s*"
        r"(?:31(?:st|nd|rd|th)?\s+March|March\s+31(?:st|nd|rd|th)?|"
        r"31(?:st|nd|rd|th)?\s+December|December\s+31(?:st|nd|rd|th)?)"
        r"[,]?\s*20\d{2}\b",
        r"\b(20\d{2}\s*[-–/]\s*(?:20)?\d{2})\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, context, re.IGNORECASE))
        if matches:
            return _clean_line(matches[-1].group(0))
    return "Not identified"


def _extract_amounts(text):
    """Find amounts while filtering dates and leaving note-number handling to the row parser."""
    text = re.sub(
        r"\(?\s*(?:refer to )?notes?\s*\d+[a-z]?\s*\)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:31(?:st|nd|rd|th)?\s+March|March\s+31(?:st|nd|rd|th)?|"
        r"31(?:st|nd|rd|th)?\s+December|December\s+31(?:st|nd|rd|th)?)"
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
        bare = re.sub(r"^(?:₹|Rs\.?|INR|USD|\$)\s*", "", raw, flags=re.IGNORECASE)
        bare = bare.strip().strip("()").replace(",", "").replace(" ", "")
        if not bare or not re.fullmatch(r"-?\d+(?:\.\d+)?", bare):
            continue
        numeric = float(bare)
        if 1900 <= numeric <= 2099 and "." not in bare:
            continue
        amounts.append((raw, bare))
    return amounts


def _row_amounts(lines, line_index, match_end):
    """Read amounts only from a likely financial table row, rejecting narrative percentages."""
    label_line = lines[line_index]
    suffix = label_line[match_end:]
    # Reject prose/growth callouts such as "REVENUE ↑ 7.9%" and rows with
    # percentages, which are not the requested currency amounts.
    if re.search(r"[↑↓%]|\b(?:growth|increase|decrease|margin|同比)\b", label_line, re.IGNORECASE):
        return [], label_line
    same_line = _extract_amounts(suffix)
    evidence_lines = [label_line]
    amounts = same_line

    if not amounts:
        following = []
        for next_line in lines[line_index + 1:line_index + 7]:
            # A numeric-only line may be a note number or a financial column.
            if re.search(r"[A-Za-z]{2,}", next_line):
                break
            if next_line.strip():
                following.append(next_line.strip())
        if following:
            amounts = _extract_amounts(" ".join(following))
            evidence_lines.extend(following)

    # Annual-report tables commonly place a note reference (e.g. 25) before
    # current-year and prior-year values. If three or more numbers exist and
    # the first is a small integer, treat it as the note reference.
    if len(amounts) >= 3:
        first = amounts[0][1]
        if re.fullmatch(r"\d{1,2}", first):
            amounts = amounts[1:]

    return amounts, " | ".join(evidence_lines)


def extract_with_ai(client, company, pages):
    # Local, rule-based extraction; no API key is needed.
    # Values are selected from labelled rows, with note references skipped.
    results = []
    seen = set()
    borrowings_by_page = {}

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        lines = [_clean_line(line) for line in page_text.splitlines()]
        lines = [line for line in lines if line]
        page_unit = _infer_unit(page_text)
        page_period = _infer_period(lines)

        for line_index, line in enumerate(lines):
            # Capture direct standard-metric labels.
            for metric, pattern in PATTERNS.items():
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    # Do not treat a sales-and-services subtotal as the standard
                    # Revenue metric when the report separately labels Revenue from Operations.
                    if metric == "Revenue" and re.search(r"\bvalue of sales\s*&\s*services\b", line, re.IGNORECASE):
                        continue
                    amounts, evidence_line = _row_amounts(lines, line_index, match.end())
                    if not amounts:
                        continue

                    context_start = max(0, line_index - 20)
                    context = " ".join(lines[context_start:min(len(lines), line_index + 7)])
                    unit = _infer_unit(context)
                    if unit == "Unit not identified":
                        unit = page_unit

                    # Preserve multiple values from a financial row rather than
                    # silently discarding the comparative column. Periods are
                    # assigned only when explicit year labels are visible nearby;
                    # otherwise the candidate is marked as needing period review.
                    nearby = " ".join(lines[max(0, line_index - 8):min(len(lines), line_index + 2)])
                    year_matches = re.findall(
                        r"\b(?:FY\s*)?(20\d{2}\s*[-–/]\s*(?:20)?\d{2}|20\d{2})\b",
                        nearby,
                        re.IGNORECASE,
                    )
                    row_periods = []
                    for year_label in year_matches:
                        normalized = _clean_line(year_label)
                        if normalized not in row_periods:
                            row_periods.append(normalized)

                    # For two numeric columns, only map them to years when two
                    # explicit year labels are present in nearby statement headers.
                    period_for_values = []
                    if len(amounts) >= 2 and len(row_periods) >= 2:
                        period_for_values = row_periods[:len(amounts)]
                    elif len(amounts) == 1:
                        period_for_values = [page_period]
                    else:
                        period_for_values = ["Period unclear"] * len(amounts)

                    for amount_index, (raw_value, value) in enumerate(amounts):
                        candidate_period = period_for_values[amount_index]
                        key = (page_num, metric, value, candidate_period, evidence_line)
                        if key in seen:
                            continue
                        seen.add(key)

                        results.append({
                            "company": company,
                            "metric": metric,
                            "reported_label": match.group(0),
                            "value": value,
                            "unit": unit,
                            "period": candidate_period,
                            "pdf_page": page_num,
                            "evidence": evidence_line[:500],
                            "confidence": (
                                "Review required"
                                if candidate_period in {"Period unclear", "Not identified"}
                                else "Review required; period label detected"
                            ),
                        })

            # Track current and non-current Borrowings rows so a total can be
            # calculated when the report does not state "Total Borrowings".
            if re.fullmatch(r"borrowings", line, re.IGNORECASE):
                amounts, evidence_line = _row_amounts(lines, line_index, len(line))
                if amounts:
                    borrowings_by_page.setdefault(page_num, []).append({
                        "value": amounts[0][1],
                        "evidence": evidence_line,
                        "unit": page_unit,
                        "period": page_period,
                    })

    # Derive total borrowings only when exactly two labelled Borrowings rows
    # occur on one page (typically non-current plus current borrowings).
    explicit_borrowings = [r for r in results if r["metric"] == "Total Borrowings"]
    if not explicit_borrowings:
        for page_num, rows in borrowings_by_page.items():
            if len(rows) != 2:
                continue
            try:
                total = sum(float(row["value"]) for row in rows)
            except ValueError:
                continue
            first = rows[0]
            results.append({
                "company": company,
                "metric": "Total Borrowings",
                "reported_label": "Borrowings (non-current + current)",
                "value": str(int(total)) if total.is_integer() else str(total),
                "unit": first["unit"],
                "period": first["period"],
                "pdf_page": page_num,
                "evidence": "Derived from two Borrowings rows: " + " | ".join(
                    row["evidence"] for row in rows
                ),
                "confidence": "Derived; verify both components",
            })

    return results

def build_summary_table(df):
    """Show one company-level row, prioritising the latest explicit reporting period."""
    columns = ["Company", "Period"] + METRICS
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    # Avoid one summary row per unrelated date found throughout the report.
    years = working["period"].astype(str).str.findall(r"20\d{2}")
    working["_period_year"] = years.apply(lambda values: max(map(int, values)) if values else 0)
    company = str(working.iloc[0].get("company", ""))
    best_year = int(working["_period_year"].max())
    current = working[working["_period_year"] == best_year] if best_year else working
    period_values = [p for p in current["period"].astype(str).unique() if p and p != "Not identified"]
    period = period_values[0] if len(period_values) == 1 else (f"Latest identified year: {best_year}" if best_year else "Not identified")

    row = {"Company": company, "Period": period}
    for metric in METRICS:
        candidates = current[current["metric"] == metric].copy()
        # Prefer explicit balance-sheet / income-statement labels over broad
        # labels, and do not display duplicates as if they were a single answer.
        if not candidates.empty:
            candidates["_label_rank"] = candidates["reported_label"].astype(str).str.lower().map(
                lambda label: 0 if label in {
                    "revenue from operations", "profit after tax", "total assets",
                    "total liabilities", "cash and cash equivalents", "total borrowings"
                } else 1
            )
            candidates = candidates.sort_values(
                ["_label_rank", "pdf_page"], ascending=[True, True]
            )
        unique = []
        for _, candidate in candidates.iterrows():
            value = str(candidate.get("value", "")).strip()
            unit = str(candidate.get("unit", "")).strip()
            if not value or value.lower() == "nan":
                continue
            if not unit or unit.lower() == "nan":
                unit = "Unit not identified"
            display = f"{value} ({unit})"
            if display not in unique:
                unique.append(display)
        if not unique:
            row[metric] = "Not found"
        elif len(unique) == 1:
            row[metric] = unique[0]
        else:
            row[metric] = "Review required: " + " | ".join(unique[:3])
    return pd.DataFrame([row], columns=columns)


def reconciliation(df):
    """Create actionable review flags without calling every repeated row a duplicate."""
    columns = [
        "Company", "Period", "Metric / Check", "Status",
        "Values found", "Source pages", "Recommended action"
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    checks = []
    # Evaluate each company and period separately. Multiple entries with the
    # same value are not conflicts; only distinct values are flagged as competing.
    for (company, period), group in df.groupby(
        ["company", "period"], dropna=False, sort=False
    ):
        found_metrics = set(group["metric"].dropna().astype(str))
        for metric in METRICS:
            candidates = group[group["metric"] == metric].copy()
            if candidates.empty:
                checks.append({
                    "Company": company,
                    "Period": period,
                    "Metric / Check": metric,
                    "Status": "Missing",
                    "Values found": "None",
                    "Source pages": "None",
                    "Recommended action": "Check whether the metric is present in the report and review its statement pages.",
                })
                continue

            values = []
            for _, candidate in candidates.iterrows():
                value = str(candidate.get("value", "")).strip()
                unit = str(candidate.get("unit", "")).strip()
                if not value or value.lower() == "nan":
                    continue
                if not unit or unit.lower() == "nan":
                    unit = "Unit not identified"
                entry = f"{value} ({unit})"
                if entry not in values:
                    values.append(entry)

            pages = sorted({
                str(int(p)) for p in candidates["pdf_page"].dropna()
                if str(p).replace(".", "", 1).isdigit()
            })
            page_text = ", ".join(pages) if pages else "Not recorded"

            if not values:
                checks.append({
                    "Company": company,
                    "Period": period,
                    "Metric / Check": metric,
                    "Status": "Missing",
                    "Values found": "No usable numeric value",
                    "Source pages": page_text,
                    "Recommended action": "Inspect the source evidence and confirm the metric label and numeric columns.",
                })
            elif len(values) > 1:
                checks.append({
                    "Company": company,
                    "Period": period,
                    "Metric / Check": metric,
                    "Status": "Conflicting candidates",
                    "Values found": " | ".join(values[:5]),
                    "Source pages": page_text,
                    "Recommended action": "Compare the reported labels, statement type, units, and period columns before choosing a value.",
                })

        # Keep one clear reminder per company-period rather than repeating a
        # generic manual-review row for every metric.
        checks.append({
            "Company": company,
            "Period": period,
            "Metric / Check": "Overall source verification",
            "Status": "Manual review",
            "Values found": "Not automatically reconciled",
            "Source pages": ", ".join(sorted({
                str(int(p)) for p in group["pdf_page"].dropna()
                if str(p).replace(".", "", 1).isdigit()
            })) or "Not recorded",
            "Recommended action": "Verify the extracted figures, reporting period, currency/unit, and evidence against the original PDF.",
        })

    return pd.DataFrame(checks, columns=columns)


with st.sidebar:
    st.header("About this tool")
    st.write(
        "Extract key financial figures from annual report PDFs, "
        "review the page evidence behind each result, and export the "
        "extracted data and review flags to Excel."
    )
    st.caption(
        "Designed for reports from different companies and financial years. "
        "Always compare results with the source report before relying on them."
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
                st.subheader("Extracted figures summary")
                st.caption(
                    "One row per company and reporting period. Each financial "
                    "metric has its own column; values include detected units. "
                    "'Not found' means no candidate was extracted, while "
                    "'Multiple candidates' means the figures need review."
                )
                summary_df = build_summary_table(df)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("View extracted details and source evidence"):
                    detail_columns = [
                        "company", "metric", "reported_label", "value",
                        "unit", "period", "pdf_page", "evidence", "confidence"
                    ]
                    st.dataframe(
                        df[detail_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

                st.subheader("Reconciliation / review flags")
                st.caption(
                    "Missing metrics and conflicting extracted values are listed separately. "
                    "Repeated identical values are not treated as conflicts. These checks "
                    "do not prove a figure is correct; compare flagged items with the source PDF."
                )
                checks = reconciliation(df)
                st.dataframe(checks, use_container_width=True, hide_index=True)

                output = io.BytesIO()

                with pd.ExcelWriter(
                    output, engine="openpyxl"
                ) as writer:
                    summary_df.to_excel(
                        writer,
                        sheet_name="Financial Summary",
                        index=False
                    )
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

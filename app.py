import io
import re
import json
import urllib.request
import urllib.error

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
    "Total Assets": r"\btotal assets\b",
    "Total Liabilities": r"\btotal liabilities\b",
    "Total Borrowings": r"\btotal borrowings\b",
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
    """Infer currency and scale only from a short, explicit unit declaration."""
    text = _clean_line(str(context or ""))
    has_inr = bool(re.search(r"\bINR\b|\bRs\.?\b|₹|\bindian rupees?\b|\brupees?\b", text, re.I))
    currencies = [
        (r"\bUSD\b|\$|\bUS dollars?\b", "USD"),
        (r"\bEUR\b|€|\beuros?\b", "EUR"),
        (r"\bGBP\b|£|\bpounds sterling\b", "GBP"),
        (r"\bAED\b", "AED"),
        (r"\bSGD\b", "SGD"),
        (r"\bJPY\b|¥", "JPY"),
        (r"\bCAD\b", "CAD"),
        (r"\bAUD\b", "AUD"),
    ]
    currency = "INR" if has_inr else next(
        (code for pattern, code in currencies if re.search(pattern, text, re.I)),
        None,
    )
    scale_checks = [
        (r"\bcrores?\b", "crore"),
        (r"\blakhs?\b", "lakh"),
        (r"\bthousands?\b|\b000s\b", "thousand"),
        (r"\bmillions?\b", "million"),
        (r"\bbillions?\b", "billion"),
    ]
    scale = next((name for pattern, name in scale_checks if re.search(pattern, text, re.I)), None)

    # Do not assume the currency from an isolated symbol or a nearby number.
    # The declaration must explicitly identify a currency, or use INR notation.
    if currency and scale:
        return f"{currency} {scale}"
    if currency and re.search(r"\b(?:currency|amounts?|figures?|values?)\b", text, re.I):
        return currency
    if currency and re.fullmatch(r"\s*(?:USD|EUR|GBP|AED|SGD|JPY|CAD|AUD|INR)\s*", text, re.I):
        return currency
    return "Unit not identified"


def _infer_statement_unit(lines, line_index, lookback=18):
    """Use the nearest explicit unit declaration above the row, within its section."""
    section_break = re.compile(
        r"^(?:notes to (?:the )?financial statements|"
        r"statement of (?:profit or loss|financial position|cash flows)|"
        r"balance sheet|income statement|cash flow statement)\b",
        re.I,
    )
    start = max(0, line_index - lookback)
    for idx in range(line_index, start - 1, -1):
        line = _clean_line(lines[idx])
        if not line:
            continue
        if idx != line_index and section_break.search(line):
            # A new statement heading starts a fresh search area; still inspect
            # that heading itself on the next iteration.
            if not re.search(r"\b(?:crores?|lakhs?|thousands?|millions?|billions?)\b", line, re.I):
                break
        has_scale = bool(re.search(r"\b(?:crores?|lakhs?|thousands?|millions?|billions?|000s)\b", line, re.I))
        declaration_like = bool(re.search(
            r"\b(?:in|amounts?|figures?|values?|expressed|denominated|currency)\b|"
            r"\b(?:INR|USD|EUR|GBP|AED|SGD|JPY|CAD|AUD)\b|₹|€|£|\$|¥|Rs\.?",
            line, re.I
        ))
        if has_scale and declaration_like:
            unit = _infer_unit(line)
            if unit != "Unit not identified":
                return unit
        # Permit explicit currency-only headings, but never infer currency from
        # ordinary financial rows containing values.
        if not has_scale and re.fullmatch(
            r"(?:amounts? (?:in|expressed in) )?(?:INR|USD|EUR|GBP|AED|SGD|JPY|CAD|AUD)",
            line, re.I
        ):
            return _infer_unit(line)
    return "Unit not identified"

def _historical_inr_rate(currency, period, metric, cache):
    """Fetch historical FX data without an API key; return INR per source currency."""
    if currency == "INR":
        return 1.0
    years = [int(y) for y in re.findall(r"20\d{2}", str(period))]
    if not years:
        return None
    end_year = max(years)
    fiscal = bool(re.search(r"FY|20\d{2}\s*[-–/]\s*(?:20)?\d{2}", str(period), re.I))
    if metric in {"Revenue", "Profit After Tax"}:
        start = f"{end_year - 1}-04-01" if fiscal else f"{end_year}-01-01"
        end = f"{end_year}-03-31" if fiscal else f"{end_year}-12-31"
        key = (currency, start, end, "average")
    else:
        end = f"{end_year}-03-31" if fiscal else f"{end_year}-12-31"
        start = end
        key = (currency, start, end, "period-end")
    if key in cache:
        return cache[key]
    url = f"https://api.frankfurter.dev/v1/{start}..{end}?from={currency}&to=INR"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AnnualReportExtractionEngine/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rates = [float(v["INR"]) for v in payload.get("rates", {}).values() if "INR" in v]
        rate = (sum(rates) / len(rates)) if rates and key[3] == "average" else (rates[-1] if rates else None)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        rate = None
    cache[key] = rate
    return rate


def _convert_to_inr_crore(value, unit, period, metric, cache):
    """Convert an extracted amount into INR crore while retaining the original."""
    try:
        amount = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None, "Conversion unavailable"
    unit = str(unit or "").strip()
    scale = 1.0
    if re.search(r"\bcrore\b", unit, re.I):
        scale = 1.0
    elif re.search(r"\blakh\b", unit, re.I):
        scale = 0.01
    elif re.search(r"\bthousand\b", unit, re.I):
        scale = 0.00001
    elif re.search(r"\bmillion\b", unit, re.I):
        scale = 0.1
    elif re.search(r"\bbillion\b", unit, re.I):
        scale = 10.0
    elif unit in {"INR", "USD", "EUR", "GBP", "AED", "SGD", "JPY", "CAD", "AUD"}:
        scale = 0.0000001
    else:
        return None, "Currency/unit not identified"
    currency_match = re.match(r"([A-Z]{3})", unit)
    currency = currency_match.group(1) if currency_match else ("INR" if unit.startswith("INR") else None)
    if currency is None:
        return None, "Currency/unit not identified"
    rate = _historical_inr_rate(currency, period, metric, cache)
    if rate is None:
        return None, "Historical rate unavailable"
    return amount * scale * rate, "Converted to INR crore"

def _period_labels(text):
    """Return distinct explicit reporting periods in the order they appear."""
    patterns = [
        r"\bFY\s*20\d{2}\s*[-–/]\s*(?:20)?\d{2}\b",
        r"\b(?:year ended|as at|as of|for the year ended)\s*"
        r"(?:31(?:st|nd|rd|th)?\s+March|March\s+31(?:st|nd|rd|th)?|"
        r"31(?:st|nd|rd|th)?\s+December|December\s+31(?:st|nd|rd|th)?)"
        r"[,]?\s*20\d{2}\b",
        r"\b20\d{2}\s*[-–/]\s*(?:20)?\d{2}\b",
        r"\b20\d{2}\b",
    ]
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            label = _clean_line(match.group(0))
            if label not in found:
                found.append(label)
        if found:
            break
    return found


def _infer_period(lines, line_index=None):
    """Return the nearest explicit reporting period, if one is visible."""
    context = " ".join(lines if line_index is None else lines[max(0, line_index - 12):line_index + 1])
    periods = _period_labels(context)
    return periods[-1] if periods else "Not identified"


def _detect_statement_type(lines, line_index, lookback=35):
    """Identify consolidated vs standalone from the nearest statement heading."""
    start = max(0, line_index - lookback)
    for idx in range(line_index, start - 1, -1):
        line = _clean_line(lines[idx])
        if re.search(r"\bconsolidated\b", line, re.I):
            return "Consolidated"
        if re.search(r"\bstandalone\b|\bstand-alone\b", line, re.I):
            return "Standalone"
        if re.search(
            r"^(?:notes to (?:the )?financial statements|"
            r"statement of (?:profit or loss|financial position|cash flows)|"
            r"balance sheet|income statement|cash flow statement)\b",
            line, re.I
        ):
            # Continue past generic statement titles, but stop at a new notes section.
            if re.search(r"^notes to", line, re.I):
                break
    return "Statement type unclear"


def _column_periods(lines, line_index, value_count):
    """Map row values to the ordered year/date headers immediately above the row."""
    start = max(0, line_index - 18)
    header_lines = []
    for idx in range(line_index - 1, start - 1, -1):
        line = _clean_line(lines[idx])
        if not line:
            continue
        # Stop if we hit another financial metric row; its years are not this row's headers.
        if any(re.search(pattern, line, re.I) for pattern in PATTERNS.values()):
            break
        labels = _period_labels(line)
        if labels:
            header_lines.append((idx, labels))
        if len(header_lines) >= 4:
            break
    if not header_lines:
        return []
    # Earlier lines usually contain the year-column headings; preserve their left-to-right order.
    header_lines.reverse()
    ordered = []
    for _, labels in header_lines:
        for label in labels:
            if label not in ordered:
                ordered.append(label)
    if len(ordered) >= value_count:
        return ordered[-value_count:]
    return ordered

def _extract_amounts(text):
    """Extract positive numeric table values; ignore percentages, dates, and dash placeholders."""
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
    # Remove percentages before parsing numbers, so 6.6% is not treated as money.
    text = re.sub(r"\(?\s*\d[\d,]*(?:\.\d+)?\s*%\s*\)?", " ", text)
    # A dash in a financial table means no value, not a negative number.
    text = re.sub(r"(?<!\w)[—–-](?!\w)", " ", text)

    number_pattern = re.compile(
        r"(?<![A-Za-z0-9/])"
        r"(?:₹\s*|Rs\.?\s*|INR\s*|USD\s*|\$\s*)?"
        r"\(?\s*\d[\d,]*(?:\.\d+)?\s*\)?"
        r"(?![A-Za-z0-9/])",
        re.IGNORECASE,
    )
    amounts = []
    for match in number_pattern.finditer(text):
        raw = match.group(0).strip()
        bare = re.sub(r"^(?:₹|Rs\.?|INR|USD|\$)\s*", "", raw, flags=re.IGNORECASE)
        bare = bare.strip().strip("()").replace(",", "").replace(" ", "")
        if not bare or not re.fullmatch(r"\d+(?:\.\d+)?", bare):
            continue
        numeric = float(bare)
        if 1900 <= numeric <= 2099 and "." not in bare:
            continue
        amounts.append((raw, bare))
    return amounts


def _row_amounts(lines, line_index, match_end):
    """Read at most two comparable values from a labelled financial row."""
    label_line = lines[line_index]
    suffix = label_line[match_end:]
    # Percentages are removed by _extract_amounts; reject explicit growth prose,
    # but allow a normal financial row that also reports a growth percentage.
    if re.search(r"\b(?:growth|increase|decrease|margin|同比)\b", label_line, re.IGNORECASE):
        return [], label_line

    amounts = _extract_amounts(suffix)
    evidence_lines = [label_line]

    if not amounts:
        # In some PDFs the label and its numeric columns are on adjacent lines.
        # Read only the first following numeric-only line to avoid swallowing
        # values from several unrelated rows below the label.
        for next_line in lines[line_index + 1:line_index + 4]:
            if not next_line.strip():
                continue
            if re.search(r"[A-Za-z]{2,}", next_line):
                break
            next_amounts = _extract_amounts(next_line)
            if next_amounts:
                amounts = next_amounts
                evidence_lines.append(next_line.strip())
                break

    # If the first small integer is a note reference, remove it before limiting
    # the row to the current and comparative values.
    if len(amounts) >= 3 and re.fullmatch(r"\d{1,2}", amounts[0][1]):
        amounts = amounts[1:]

    # This prototype only retains the two most plausible table columns.
    # Extra numbers are not silently treated as additional financial periods.
    amounts = amounts[:2]
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
        page_period = _infer_period(lines)

        for line_index, line in enumerate(lines):
            # Capture direct standard-metric labels.
            for metric, pattern in PATTERNS.items():
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    # Require the metric label to begin the row (allowing common
                    # table markers) so narrative mentions are not treated as figures.
                    row_prefix = line[:match.start()].strip(" •|:-–—")
                    if row_prefix:
                        continue
                    # Do not treat a sales-and-services subtotal as the standard
                    # Revenue metric when the report separately labels Revenue from Operations.
                    if metric == "Revenue" and re.search(r"\bvalue of sales\s*&\s*services\b", line, re.IGNORECASE):
                        continue
                    amounts, evidence_line = _row_amounts(lines, line_index, match.end())
                    if not amounts:
                        continue

                    # Read currency and scale from an explicit declaration above
                    # this row in the same statement section. Never fall back to
                    # unrelated text elsewhere on the page.
                    unit = _infer_statement_unit(lines, line_index)

                    # Map each extracted amount to the actual ordered year/date
                    # headers above this row, not any year mentioned nearby in prose.
                    statement_type = _detect_statement_type(lines, line_index)
                    column_periods = _column_periods(lines, line_index, len(amounts))
                    if len(amounts) == 1:
                        period_for_values = [column_periods[0] if column_periods else page_period]
                    elif len(column_periods) >= len(amounts):
                        period_for_values = column_periods[:len(amounts)]
                    else:
                        period_for_values = ["Period unclear"] * len(amounts)

                    for amount_index, (raw_value, value) in enumerate(amounts):
                        candidate_period = (
                            period_for_values[amount_index]
                            if amount_index < len(period_for_values)
                            else "Period unclear"
                        )
                        key = (page_num, metric, value, candidate_period, statement_type, evidence_line)
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
                            "statement_type": statement_type,
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
                        "unit": _infer_statement_unit(lines, line_index),
                        "period": page_period,
                        "statement_type": _detect_statement_type(lines, line_index),
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
                "statement_type": first.get("statement_type", "Statement type unclear"),
                "pdf_page": page_num,
                "evidence": "Derived from two Borrowings rows: " + " | ".join(
                    row["evidence"] for row in rows
                ),
                "confidence": "Derived; verify both components",
            })

    fx_cache = {}
    for item in results:
        converted, status = _convert_to_inr_crore(
            item.get("value"), item.get("unit"), item.get("period"),
            item.get("metric"), fx_cache
        )
        item["value_inr_crore"] = round(converted, 6) if converted is not None else None
        item["conversion_status"] = status

    return results

def build_summary_table(df):
    """Show one selected figure per metric, with one shared reporting-period column."""
    columns = ["Company", "Statement Type", "Period"] + METRICS
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    years = working["period"].astype(str).str.findall(r"20\d{2}")
    working["_period_year"] = years.apply(
        lambda values: max(map(int, values)) if values else 0
    )
    company = str(working.iloc[0].get("company", ""))

    preferred_labels = {
        "Revenue": [
            "revenue from operations",
            "revenue from contracts with customers",
            "total revenue",
        ],
        "Profit After Tax": [
            "profit after tax",
            "profit for the year attributable to owners",
            "net profit after tax",
        ],
        "Total Assets": ["total assets"],
        "Total Liabilities": ["total liabilities"],
        "Total Borrowings": ["total borrowings"],
        "Cash and Cash Equivalents": ["cash and cash equivalents"],
    }

    statement_types = (
        working["statement_type"].fillna("Statement type unclear").astype(str).drop_duplicates().tolist()
        if "statement_type" in working.columns else ["Statement type unclear"]
    )
    summary_rows = []
    for statement_type in statement_types:
        row = {"Company": company, "Statement Type": statement_type}
        selected_periods = []
        for metric in METRICS:
            candidates = working[working["metric"] == metric].copy()
            if "statement_type" in candidates.columns:
                candidates = candidates[
                    candidates["statement_type"].fillna("Statement type unclear").astype(str) == statement_type
                ]
        if candidates.empty:
            row[metric] = "Not found"
            continue

        candidates["_label_rank"] = candidates["reported_label"].astype(str).str.lower().map(
            lambda label: (
                preferred_labels[metric].index(label)
                if label in preferred_labels[metric]
                else len(preferred_labels[metric])
            )
        )
        candidates["_period_rank"] = candidates["period"].astype(str).map(
            lambda value: 1 if value in {"Period unclear", "Not identified"} else 0
        )
        candidates["_unit_rank"] = candidates["unit"].astype(str).map(
            lambda unit: 1 if unit in {"Unit not identified", "", "nan"} else 0
        )
        candidates = candidates.sort_values(
            ["_period_rank", "_period_year", "_label_rank", "_unit_rank", "pdf_page"],
            ascending=[True, False, True, True, True],
            kind="stable",
        )

        selected = candidates.iloc[0]
        converted = selected.get("value_inr_crore")
        period = str(selected.get("period", "")).strip()

        if pd.isna(converted):
            row[metric] = "Conversion unavailable; review"
        else:
            row[metric] = f"₹ {converted:,.2f} crore"

        if period and period.lower() != "nan" and period not in {
            "Period unclear", "Not identified"
        }:
            selected_periods.append(period)

        unique_periods = list(dict.fromkeys(selected_periods))
        if len(unique_periods) == 1:
            row["Period"] = unique_periods[0]
        elif len(unique_periods) > 1:
            row["Period"] = "Mixed periods; review required"
        else:
            row["Period"] = "Not identified"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=columns)

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
        "Summary values are shown in INR crore. Foreign-currency conversion uses historical rates when available and flags unavailable rates or units for review. Always verify results against the source report."
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
                    "All summary figures are standardised to INR crore. Foreign-currency figures use historical rates when available. Revenue/profit use an annual average rate; balance-sheet figures use a period-end rate. Original values and units remain in the details. Review unavailable conversions against the source PDF."
                )
                summary_df = build_summary_table(df)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("View extracted details and source evidence"):
                    detail_columns = [
                        "company", "statement_type", "metric", "reported_label", "value",
                        "unit", "value_inr_crore", "conversion_status", "period", "pdf_page", "evidence", "confidence"
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

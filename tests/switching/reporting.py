"""Generate the switching Excel workbook from actual pytest outcomes."""

from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
import math
import statistics
import zipfile


RESULT_HEADERS = [
    "Test Case ID", "Test Case Name", "Expected Scenario", "Result / Failure Reason",
    "Test Type", "Expected Outcome", "Actual Outcome", "Test Status",
    "Outcome Validity", "Switch From", "Switch To", "Switch Time",
    "First Transcript Time", "Total Latency", "2-Second Target",
    "First Transcript", "Final Transcript", "Notes",
]
FAILURE_HEADERS = [
    "Test ID", "Test Name", "Category", "Expected", "Actual", "Failure Reason",
    "Latency", "Exception", "Suspected Component", "Log Path",
    "Recommended Investigation",
]


def percentile95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _validity(status):
    return "EXPECTED" if status in {"PASS", "XFAIL"} else "UNEXPECTED"


def _column_name(index):
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(65 + remainder) + value
    return value


def _sheet_xml(rows):
    rendered = []
    for row_index, row in enumerate(rows, 1):
        cells = []
        for column_index, value in enumerate(row, 1):
            if value is None:
                continue
            reference = f"{_column_name(column_index)}{row_index}"
            style = ' s="1"' if row_index == 1 else ""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{reference}"{style}><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(
                    f'<c r="{reference}" t="inlineStr"{style}><is><t>{text}</t></is></c>')
        rendered.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(rendered)}</sheetData></worksheet>')


def _write_xlsx(path, sheets):
    """Write a standards-compliant XLSX using OOXML and the standard library."""
    content_types = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append('</Types>')
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _rows) in enumerate(sheets, 1))
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1))
    relationships += (
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>')
    styles = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font/><font><b/><color rgb="FFFFFFFF"/></font></fonts>'
              '<fills count="3"><fill><patternFill patternType="none"/></fill>'
              '<fill><patternFill patternType="gray125"/></fill>'
              '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/>'
              '<bgColor indexed="64"/></patternFill></fill></fills>'
              '<borders count="1"><border/></borders>'
              '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
              '<cellXfs count="2"><xf/><xf fontId="1" fillId="2" applyFont="1" applyFill="1"/></cellXfs>'
              '</styleSheet>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                         '</Relationships>')
        archive.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8"?>'
                         '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                         'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                         f'<sheets>{workbook_sheets}</sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         f'{relationships}</Relationships>')
        archive.writestr("xl/styles.xml", styles)
        for index, (_name, rows) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))


def write_report(records, output_directory="test_reports"):
    """Write one timestamped workbook and return its path."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / datetime.now().strftime("switching_test_report_%Y%m%d_%H%M%S_%f.xlsx")
    result_rows = [RESULT_HEADERS]
    for record in records:
        item, status = record["evidence"], record["status"]
        latency = item.total_latency
        target = "N/A" if latency is None else "PASS" if latency <= 2.0 else "FAIL"
        result_rows.append([
            item.test_id, item.name, item.expected_scenario,
            record.get("exception") or item.reason, item.test_type, item.expected_outcome,
            item.actual_outcome, status, _validity(status), item.switch_from, item.switch_to,
            item.switch_time, item.first_transcript_time, latency, target,
            item.first_transcript, item.final_transcript, item.notes,
        ])

    runs = [run for record in records for run in record["evidence"].latency_runs]
    latencies = [run["total_latency"] for run in runs if run.get("total_latency") is not None]
    performance_rows = [["Metric", "Value"],
        ["Test count", len(records)],
        ["Successful switches", sum(r["status"] == "PASS" for r in records)],
        ["Failed switches", sum(r["status"] in {"FAIL", "ERROR", "XPASS"} for r in records)],
        ["Minimum latency", min(latencies) if latencies else None],
        ["Maximum latency", max(latencies) if latencies else None],
        ["Average latency", statistics.fmean(latencies) if latencies else None],
        ["Median latency", statistics.median(latencies) if latencies else None],
        ["P95 latency", percentile95(latencies)],
        ["Under 2 seconds", sum(value <= 2.0 for value in latencies)],
        ["Over 2 seconds", sum(value > 2.0 for value in latencies)],
    ]
    for source, target in (("Hindi", "English"), ("English", "Hindi"),
                           ("Hinglish", "English"), ("English", "Hinglish")):
        values = [run["total_latency"] for run in runs
                  if run.get("switch_from") == source and run.get("switch_to") == target]
        performance_rows.append([f"{source} → {target} average",
                                 statistics.fmean(values) if values else None])

    failure_rows = [FAILURE_HEADERS]
    for record in records:
        if record["status"] not in {"FAIL", "ERROR", "XFAIL", "XPASS"}:
            continue
        item = record["evidence"]
        failure_rows.append([
            item.test_id, item.name, item.test_type, item.expected_outcome,
            item.actual_outcome, record.get("exception") or item.reason,
            item.total_latency, record.get("exception", ""), item.suspected_component,
            record.get("log_path", ""), item.recommended_investigation,
        ])

    summary_rows = [["Level", "Passed", "Failed", "XFail", "Error"]]
    for level in ("LIGHT", "STATE", "PERF", "COMPLEX", "NEG", "STRESS"):
        selected = [r["status"] for r in records if level in r["evidence"].test_id]
        counts = Counter(selected)
        summary_rows.append([level, counts["PASS"], counts["FAIL"] + counts["XPASS"],
                             counts["XFAIL"], counts["ERROR"]])

    latency_rows = [["Test ID", "Run", "From", "To", "T0", "T1", "T2", "T3", "T4",
                     "Switch Config", "First Text", "UI Delay", "Total Latency"]]
    for record in records:
        for run in record["evidence"].latency_runs:
            latency_rows.append([record["evidence"].test_id, *[run.get(key) for key in (
                "run", "switch_from", "switch_to", "t0", "t1", "t2", "t3", "t4",
                "switch_config", "first_text", "ui_delay", "total_latency")]])
    _write_xlsx(path, [("Test Results", result_rows),
                       ("Performance Summary", performance_rows),
                       ("Failure Summary", failure_rows), ("Summary", summary_rows),
                       ("Latency Runs", latency_rows)])
    return path

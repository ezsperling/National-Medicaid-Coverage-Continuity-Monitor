"""Download and normalize public CMS state-month data (50 states and DC)."""
import csv
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
BASE = "https://data.medicaid.gov/api/1/datastore/query/{}/0/download?format=csv"
SOURCES = {
    "enrollment": "6165f45b-ca93-5bb5-9d06-db29c692a360",
    "renewals": "5abea2e0-3f8e-4b49-a50d-d63d5fd9103c",
}
STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


def fetch(dataset_id):
    request = urllib.request.Request(BASE.format(dataset_id), headers={"User-Agent": "NationalMedicaidMonitor/0.1 research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))


def number(value):
    value = (value or "").strip().replace(",", "")
    if not value or value.upper() in ("N/A", "NA", "NULL", "-"):
        return ""
    try:
        return float(value)
    except ValueError:
        return ""


def select(rows, version_col, preferred):
    selected = {}
    for row in rows:
        state, month = row["State Abbreviation"], row["Reporting Period"]
        if state not in STATES or len(month) != 6 or not month.isdigit():
            continue
        key = (state, month)
        if key not in selected or (row.get(version_col) == preferred and selected[key].get(version_col) != preferred):
            selected[key] = row
    return selected


def main():
    raw_e = fetch(SOURCES["enrollment"])
    raw_r = fetch(SOURCES["renewals"])
    enroll = select(raw_e, "Preliminary or Updated", "U")
    renew = select(raw_r, "Original or Updated", "U")
    columns = ["state", "state_name", "month", "expanded", "enrollment", "medicaid_enrollment", "chip_enrollment", "adult_enrollment", "call_volume", "wait_minutes", "abandon_rate", "renewal_due", "renewed", "ex_parte", "procedural", "ineligible", "pending", "enrollment_version", "renewal_version", "enrollment_footnote", "medicaid_footnote", "chip_footnote", "renewal_due_footnote", "renewal_footnote", "ex_parte_footnote"]
    output = []
    for key in sorted(set(enroll) | set(renew), key=lambda x: (x[1], x[0])):
        e, r = enroll.get(key, {}), renew.get(key, {})
        output.append({
            "state": key[0], "state_name": e.get("State Name") or r.get("State Name") or key[0], "month": key[1],
            "expanded": e.get("State Expanded Medicaid", ""),
            "enrollment": number(e.get("Total Medicaid and CHIP Enrollment")),
            "medicaid_enrollment": number(e.get("Total Medicaid Enrollment")),
            "chip_enrollment": number(e.get("Total CHIP Enrollment")),
            "adult_enrollment": number(e.get("Total Adult Medicaid Enrollment")),
            "call_volume": number(e.get("Total Call Center Volume (Number of Calls)")),
            "wait_minutes": number(e.get("Average Call Center Wait Time (Minutes)")),
            "abandon_rate": number(e.get("Average Call Center Abandonment Rate")),
            "renewal_due": number(r.get("Beneficiaries with a Renewal Due")),
            "renewed": number(r.get("Beneficiaries Whose Coverage Was Renewed (Total)")),
            "ex_parte": number(r.get("Beneficiaries Whose Coverage Was Renewed on an Ex Parte Basis")),
            "procedural": number(r.get("Beneficiaries Disenrolled for Procedural Reasons at Renewal")),
            "ineligible": number(r.get("Beneficiaries Determined Ineligible at Renewal")),
            "pending": number(r.get("Beneficiaries with a Pending Renewal")),
            "enrollment_version": e.get("Preliminary or Updated", ""),
            "renewal_version": r.get("Original or Updated", ""),
            "enrollment_footnote": e.get("Total Medicaid and CHIP Enrollment - footnotes", ""),
            "medicaid_footnote": e.get("Total Medicaid Enrollment - footnotes", ""),
            "chip_footnote": e.get("Total CHIP Enrollment - footnotes", ""),
            "renewal_due_footnote": r.get("Beneficiaries with a Renewal Due - footnotes", ""),
            "renewal_footnote": r.get("Beneficiaries Disenrolled for Procedural Reasons at Renewal - footnotes", ""),
            "ex_parte_footnote": r.get("Beneficiaries Whose Coverage Was Renewed on an Ex Parte Basis - footnotes", ""),
        })
    path = DATA / "cms_state_month.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(output)
    # Keep original and updated cohorts side by side for a transparent revision check.
    versions = {}
    for row in raw_r:
        state, month, version = row.get("State Abbreviation"), row.get("Reporting Period"), row.get("Original or Updated")
        if state in STATES and version in ("O", "U") and len(month or "") == 6:
            versions[(state, month, version)] = row
    audit_columns = ["state", "month", "original_due", "updated_due", "original_procedural", "updated_procedural", "original_pending", "updated_pending", "original_auto_renewed", "updated_auto_renewed"]
    audit = []
    for state, month in sorted({(state, month) for state, month, _ in versions}):
        original, updated = versions.get((state, month, "O"), {}), versions.get((state, month, "U"), {})
        audit.append({"state": state, "month": month,
            "original_due": number(original.get("Beneficiaries with a Renewal Due")),
            "updated_due": number(updated.get("Beneficiaries with a Renewal Due")),
            "original_procedural": number(original.get("Beneficiaries Disenrolled for Procedural Reasons at Renewal")),
            "updated_procedural": number(updated.get("Beneficiaries Disenrolled for Procedural Reasons at Renewal")),
            "original_pending": number(original.get("Beneficiaries with a Pending Renewal")),
            "updated_pending": number(updated.get("Beneficiaries with a Pending Renewal")),
            "original_auto_renewed": number(original.get("Beneficiaries Whose Coverage Was Renewed on an Ex Parte Basis")),
            "updated_auto_renewed": number(updated.get("Beneficiaries Whose Coverage Was Renewed on an Ex Parte Basis"))})
    with (DATA / "renewal_revisions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, audit_columns)
        writer.writeheader()
        writer.writerows(audit)
    common_months = sorted({month for _, month in enroll} & {month for _, month in renew})
    latest = next((m for m in reversed(common_months) if sum((s, m) in enroll and (s, m) in renew for s in STATES) == 51), None)
    metadata = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source_ids": SOURCES, "rows": len(output), "latest_joint_51_state_month": latest, "states_in_latest": sum((s, latest) in enroll and (s, latest) in renew for s in STATES) if latest else 0}
    (DATA / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

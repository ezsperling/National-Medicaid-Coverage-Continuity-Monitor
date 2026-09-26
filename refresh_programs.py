"""Fetch CMS Medicare monthly and Medicaid MLTSS annual histories."""

import csv
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
MEDICARE_ID = "d7fabe1e-d19b-4333-9eff-e80e0643f2fd"
MLTSS_ID = "5394bcab-c748-5e4b-af07-b5bf77ed3aa3"
STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MedicaidCoverageMonitor/0.2 research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def numeric(value):
    value = (value or "").replace(",", "").strip()
    return value if value.isdigit() else ""


def write_csv(name, fields, records):
    with (DATA / name).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    # Match the bundled Medicaid/CHIP snapshot month. CMS API returns state totals
    # without duplicate county rows when BENE_GEO_LVL=State.
    months = {"01": "January", "02": "February", "03": "March", "04": "April", "05": "May", "06": "June", "07": "July", "08": "August", "09": "September", "10": "October", "11": "November", "12": "December"}
    with (DATA / "metadata.json").open(encoding="utf-8") as f:
        month = json.load(f)["latest_joint_51_state_month"]
    params = urllib.parse.urlencode({"filter[YEAR]": month[:4], "filter[MONTH]": months[month[4:]], "filter[BENE_GEO_LVL]": "State", "size": 100})
    medicare_url = f"https://data.cms.gov/data-api/v1/dataset/{MEDICARE_ID}/data?{params}"
    raw = json.loads(get(medicare_url))
    medicare = [{"state": r["BENE_STATE_ABRVTN"], "month": month,
                 "medicare_enrollment": numeric(r.get("TOT_BENES")),
                 "medicare_advantage_other": numeric(r.get("MA_AND_OTH_BENES")),
                 "medicare_medicaid_duals": numeric(r.get("DUAL_TOT_BENES"))}
                for r in raw if r["BENE_STATE_ABRVTN"] in STATES]
    if len(medicare) != 51 or len({r["state"] for r in medicare}) != 51:
        raise ValueError(f"Expected 51 Medicare state rows; received {len(medicare)}")
    write_csv("medicare_state.csv", list(medicare[0]), medicare)

    # Monthly state rows for the latest 36 months, excluding the annual "Year" rows.
    start_year = int(month[:4]) - 3
    historical = []
    reverse_months = {name: number for number, name in months.items()}
    for year in range(start_year, int(month[:4]) + 1):
        query = urllib.parse.urlencode({"filter[YEAR]": year, "filter[BENE_GEO_LVL]": "State", "size": 1000})
        for row in json.loads(get(f"https://data.cms.gov/data-api/v1/dataset/{MEDICARE_ID}/data?{query}")):
            if row["BENE_STATE_ABRVTN"] not in STATES or row["MONTH"] not in reverse_months:
                continue
            report_month = str(year) + reverse_months[row["MONTH"]]
            if report_month > month or report_month < (str(start_year) + month[4:]):
                continue
            historical.append({"state": row["BENE_STATE_ABRVTN"], "month": report_month,
                               "medicare_enrollment": numeric(row.get("TOT_BENES")),
                               "medicare_advantage_other": numeric(row.get("MA_AND_OTH_BENES")),
                               "medicare_medicaid_duals": numeric(row.get("DUAL_TOT_BENES"))})
    if len({(r["state"], r["month"]) for r in historical}) != len(historical):
        raise ValueError("Duplicate Medicare state-month in CMS API")
    write_csv("medicare_history.csv", list(medicare[0]), historical)

    mltss_url = f"https://data.medicaid.gov/api/1/datastore/query/{MLTSS_ID}/0/download?format=csv"
    rows = list(csv.DictReader(io.StringIO(get(mltss_url).decode("utf-8-sig"))))
    year = max(r["Year"] for r in rows if r["Year"].isdigit())
    # Preserve source state names: footnote suffixes and missing cells need to
    # remain visible rather than invent a zero or silently map the wrong state.
    mltss = [{"state_name_source": r["State"], "year": year,
              "comprehensive_mltss": numeric(r["Comprehensive Managed Care LTSS Enrollees"]),
              "mltss_only": numeric(r["Managed LTSS Only Enrollees"]),
              "notes": r["Notes"]}
             for r in rows if r["Year"] == year and r["State"] != "TOTALS"]
    write_csv("mltss_annual.csv", list(mltss[0]), mltss)
    annual_history = [{"state_name_source": r["State"], "year": r["Year"],
                       "comprehensive_mltss": numeric(r["Comprehensive Managed Care LTSS Enrollees"]),
                       "mltss_only": numeric(r["Managed LTSS Only Enrollees"]), "notes": r["Notes"]}
                      for r in rows if r["Year"].isdigit() and r["State"] != "TOTALS"]
    write_csv("mltss_history.csv", list(mltss[0]), annual_history)
    (DATA / "program_metadata.json").write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "medicare_month": month, "medicare_dataset_id": MEDICARE_ID,
        "medicare_api_url": medicare_url, "medicare_state_rows": len(medicare),
        "mltss_year": year, "mltss_dataset_id": MLTSS_ID,
        "medicare_history_rows": len(historical), "medicare_history_start": str(start_year) + month[4:],
        "mltss_source_url": mltss_url, "mltss_rows": len(mltss),
        "mltss_history_rows": len(annual_history)}, indent=2) + "\n")
    print(f"Medicare: {month}, {len(medicare)} states, {len(historical)} historical rows; "
          f"MLTSS: {year}, {len(annual_history)} historical rows")


if __name__ == "__main__":
    main()

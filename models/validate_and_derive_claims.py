"""
Outpatient claims: (1) validate data and add validation flags, (2) add derived columns.
Function 1: validate_data() — applies cleansing rules, adds validation flag columns.
Function 2: add_derived_columns() — adds derived columns (has_negative_payment, diagnosis_is_other, etc.).
Writes new file: outpatient_claims_validated_and_derived.csv
"""
import csv
import os
import re

EDA = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(EDA, "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv")
VALIDATED_CSV = os.path.join(EDA, "outpatient_claims_validated.csv")
OUTPUT_CSV = os.path.join(EDA, "outpatient_claims_validated_and_derived.csv")

DESYNPUF_ID_PATTERN = re.compile(r"^[0-9A-Fa-f]{16}$")
YYYYMMDD_PATTERN = re.compile(r"^\d{8}$")
NPI_PATTERN = re.compile(r"^\d{10}$")


def _strip(s):
    return (s or "").strip()


def _parse_yyyymmdd(s):
    s = _strip(s)
    if not s or not YYYYMMDD_PATTERN.match(s):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_amount(s):
    s = _strip(s)
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _in_study_window(from_dt, thru_dt):
    """1 if both in 2008-2010, else 0."""
    if from_dt is None and thru_dt is None:
        return ""
    low, high = 20080101, 20101231
    from_ok = from_dt is None or (low <= from_dt <= high)
    thru_ok = thru_dt is None or (low <= thru_dt <= high)
    return "1" if (from_ok and thru_ok) else "0"


def validate_data(input_path, output_path):
    """
    Function 1: Validate each column per cleansing rules and add validation flag columns.
    Reads input_path, writes output_path with original columns + validation flags.
    Returns (output_path, list of validation column names).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    validation_columns = [
        "DESYNPUF_ID_missing", "DESYNPUF_ID_invalid", "DESYNPUF_ID_original",
        "CLM_ID_missing", "CLM_ID_invalid",
        "SEGMENT_invalid",
        "CLM_FROM_DT_invalid", "CLM_THRU_DT_invalid", "date_order_invalid", "in_study_window",
        "PRVDR_NUM_invalid",
        "CLM_PMT_AMT_invalid",
        "AT_PHYSN_NPI_invalid", "OP_PHYSN_NPI_invalid", "OT_PHYSN_NPI_invalid",
    ]

    with open(input_path, "r", encoding="utf-8", errors="replace") as f_in:
        r = csv.reader(f_in)
        header = next(r)
        ncols = len(header)
        idx = {}
        for name in [
            "DESYNPUF_ID", "CLM_ID", "SEGMENT", "CLM_FROM_DT", "CLM_THRU_DT",
            "PRVDR_NUM", "CLM_PMT_AMT", "AT_PHYSN_NPI", "OP_PHYSN_NPI", "OT_PHYSN_NPI"
        ]:
            if name in header:
                idx[name] = header.index(name)
            else:
                idx[name] = -1

        out_header = header + validation_columns
        row_count = 0

        with open(output_path, "w", newline="") as f_out:
            w = csv.writer(f_out)
            w.writerow(out_header)
            for row in r:
                row_count += 1
                if len(row) < ncols:
                    row = row + [""] * (ncols - len(row))

                # Pad row to ncols
                while len(row) < ncols:
                    row.append("")
                vals = list(row)

                def get(name):
                    i = idx.get(name, -1)
                    return vals[i] if 0 <= i < len(vals) else ""

                # --- DESYNPUF_ID ---
                raw_id = _strip(get("DESYNPUF_ID"))
                if not raw_id:
                    desynpuf_clean, miss, inv, orig = "", "1", "0", ""
                elif len(raw_id) != 16 or not DESYNPUF_ID_PATTERN.match(raw_id):
                    desynpuf_clean, miss, inv, orig = "", "0", "1", raw_id
                else:
                    desynpuf_clean, miss, inv, orig = raw_id, "0", "0", ""
                if idx.get("DESYNPUF_ID", -1) >= 0:
                    vals[idx["DESYNPUF_ID"]] = desynpuf_clean
                flags = [miss, inv, orig]

                # --- CLM_ID ---
                clm = _strip(get("CLM_ID"))
                clm_miss = "1" if not clm else "0"
                clm_inv = "0" if not clm else ("1" if not clm.isdigit() or len(clm) > 20 else "0")
                flags.extend([clm_miss, clm_inv])

                # --- SEGMENT ---
                seg = _strip(get("SEGMENT"))
                seg_inv = "1" if seg not in ("1", "2") else "0"
                flags.append(seg_inv)

                # --- CLM_FROM_DT, CLM_THRU_DT ---
                from_dt = _parse_yyyymmdd(get("CLM_FROM_DT"))
                thru_dt = _parse_yyyymmdd(get("CLM_THRU_DT"))
                from_inv = "1" if (get("CLM_FROM_DT") and from_dt is None) else "0"
                thru_inv = "1" if (get("CLM_THRU_DT") and thru_dt is None) else "0"
                order_inv = "1" if (from_dt is not None and thru_dt is not None and from_dt > thru_dt) else "0"
                study = _in_study_window(from_dt, thru_dt)
                flags.extend([from_inv, thru_inv, order_inv, study])

                # --- PRVDR_NUM ---
                prv = _strip(get("PRVDR_NUM"))
                prv_inv = "1" if (prv and len(prv) != 6) else "0"
                flags.append(prv_inv)

                # --- CLM_PMT_AMT ---
                amt = _parse_amount(get("CLM_PMT_AMT"))
                amt_inv = "1" if (get("CLM_PMT_AMT") and amt is None) else "0"
                flags.append(amt_inv)

                # --- NPI (10-digit numeric) ---
                def npi_inv(v):
                    v = _strip(v)
                    if not v:
                        return "0"
                    return "1" if not NPI_PATTERN.match(v) else "0"
                flags.append(npi_inv(get("AT_PHYSN_NPI")))
                flags.append(npi_inv(get("OP_PHYSN_NPI")))
                flags.append(npi_inv(get("OT_PHYSN_NPI")))

                w.writerow(vals + flags)

    print("validate_data(): wrote", row_count, "rows to", output_path)
    return output_path, validation_columns


def add_derived_columns(validated_path, output_path):
    """
    Function 2: Add derived columns to validated data.
    Reads validated_path (output of validate_data), adds derived columns, writes output_path.
    Returns output_path.
    """
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Validated file not found: {validated_path}")

    derived_columns = [
        "has_negative_payment",
        "diagnosis_is_other",
        "has_primary_diagnosis",
        "has_primary_procedure",
    ]

    with open(validated_path, "r", encoding="utf-8", errors="replace") as f_in:
        r = csv.reader(f_in)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        for name in ["CLM_PMT_AMT", "ICD9_DGNS_CD_1", "ICD9_PRCDR_CD_1", "HCPCS_CD_1"]:
            if name not in idx:
                idx[name] = -1

        out_header = header + derived_columns
        row_count = 0

        with open(output_path, "w", newline="") as f_out:
            w = csv.writer(f_out)
            w.writerow(out_header)
            for row in r:
                row_count += 1
                n = len(header)
                if len(row) < n:
                    row = row + [""] * (n - len(row))

                def get(name):
                    i = idx.get(name, -1)
                    return (row[i] if 0 <= i < len(row) else "").strip()

                # has_negative_payment: 1 if CLM_PMT_AMT < 0
                amt = get("CLM_PMT_AMT")
                try:
                    has_neg = "1" if float(amt.replace(",", "")) < 0 else "0"
                except (ValueError, TypeError):
                    has_neg = "0"
                # diagnosis_is_other: 1 if ICD9_DGNS_CD_1 == "OTHER"
                diag1 = get("ICD9_DGNS_CD_1")
                diag_other = "1" if diag1.upper() == "OTHER" else "0"
                # has_primary_diagnosis: 1 if ICD9_DGNS_CD_1 non-empty
                has_prim_diag = "1" if diag1 else "0"
                # has_primary_procedure: 1 if ICD9_PRCDR_CD_1 or HCPCS_CD_1 non-empty
                proc1 = get("ICD9_PRCDR_CD_1")
                hcpcs1 = get("HCPCS_CD_1")
                has_prim_proc = "1" if (proc1 or hcpcs1) else "0"

                w.writerow(row + [has_neg, diag_other, has_prim_diag, has_prim_proc])

    print("add_derived_columns(): wrote", row_count, "rows to", output_path)
    return output_path


def main():
    if not os.path.exists(INPUT_CSV):
        print("Input not found:", INPUT_CSV)
        return 1
    validate_data(INPUT_CSV, VALIDATED_CSV)
    add_derived_columns(VALIDATED_CSV, OUTPUT_CSV)
    print("New file:", OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    exit(main())

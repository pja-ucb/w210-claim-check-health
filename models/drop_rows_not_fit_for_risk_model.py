"""
Drop rows not fit for risk model.
Reads outpatient_claims_final_cleansed.csv and keeps only rows that pass all
"fit for model" criteria; writes outpatient_claims_model_ready.csv.
Reports how many rows dropped and why.
"""
import csv
import os

EDA = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(EDA, "outpatient_claims_final_cleansed.csv")
OUTPUT_CSV = os.path.join(EDA, "outpatient_claims_model_ready.csv")


def _get(row, idx, name):
    i = idx.get(name, -1)
    return (row[i] if 0 <= i < len(row) else "").strip()


def is_fit_for_risk_model(row, idx):
    """
    Return True if row is fit for risk model, False otherwise.
    Drop reasons: missing/invalid beneficiary or claim ID, invalid segment,
    invalid dates, date order wrong, or outside study window.
    """
    # Beneficiary ID required for risk (join to beneficiary, aggregate by bene)
    if _get(row, idx, "DESYNPUF_ID_missing") == "1":
        return False, "DESYNPUF_ID_missing"
    if _get(row, idx, "DESYNPUF_ID_invalid") == "1":
        return False, "DESYNPUF_ID_invalid"
    if not _get(row, idx, "DESYNPUF_ID"):
        return False, "DESYNPUF_ID_empty"

    # Claim ID required
    if _get(row, idx, "CLM_ID_missing") == "1":
        return False, "CLM_ID_missing"
    if _get(row, idx, "CLM_ID_invalid") == "1":
        return False, "CLM_ID_invalid"
    if not _get(row, idx, "CLM_ID"):
        return False, "CLM_ID_empty"

    # Valid segment (1 or 2)
    if _get(row, idx, "SEGMENT_invalid") == "1":
        return False, "SEGMENT_invalid"

    # Valid dates for service period (needed for risk features)
    if _get(row, idx, "CLM_FROM_DT_invalid") == "1":
        return False, "CLM_FROM_DT_invalid"
    if _get(row, idx, "CLM_THRU_DT_invalid") == "1":
        return False, "CLM_THRU_DT_invalid"
    if _get(row, idx, "date_order_invalid") == "1":
        return False, "date_order_invalid"

    # Within study window (2008-2010) for consistent modeling
    if _get(row, idx, "in_study_window") != "1":
        return False, "in_study_window"

    return True, None


def main():
    if not os.path.exists(INPUT_CSV):
        print("Input not found:", INPUT_CSV)
        print("Run cleanse_validated_claims.py first.")
        return 1

    with open(INPUT_CSV, "r", encoding="utf-8", errors="replace") as f_in:
        r = csv.reader(f_in)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        drop_reasons = {}
        kept = 0
        dropped = 0
        with open(OUTPUT_CSV, "w", newline="") as f_out:
            w = csv.writer(f_out)
            w.writerow(header)
            for row in r:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                fit, reason = is_fit_for_risk_model(row, idx)
                if fit:
                    w.writerow(row)
                    kept += 1
                else:
                    dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    total = kept + dropped
    print("Rows not fit for risk model — dropped.")
    print("  Input:     ", INPUT_CSV)
    print("  Output:    ", OUTPUT_CSV)
    print("  Total read:", total)
    print("  Kept:      ", kept)
    print("  Dropped:   ", dropped)
    if dropped:
        print("  Drop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print("    ", reason, ":", count)
    return 0


if __name__ == "__main__":
    exit(main())

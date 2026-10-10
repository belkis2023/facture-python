import re
from datetime import datetime
from pathlib import Path


def extract_invoice_number(result):
    #we're just keeping the text (we don't need the position nor the confidence)
    texts = [detection[1] for detection in result]

    for i, text in enumerate(texts):
        text_lower = text.lower().strip()

        # Type 1: "no: 77833987"
        if text_lower.startswith("no:"):
            return text.split(":", 1)[1].strip().lstrip(":").strip()

        # Type 2: "Invoice#t: 121"
        if text_lower.startswith("invoice#"):
            return text.split(":", 1)[1].strip().lstrip(":").strip()

        # Type 3: "INVOICE NO." followed by "101"
        if text_lower == "invoice no.":
            if i + 1 < len(texts):
                return texts[i + 1].strip()

    return None

def _get_texts(result):
    # EasyOCR returns detections as (box, text, confidence) tuples.
    return [detection[1].strip() for detection in result]


def extract_supplier(result):
    texts = _get_texts(result)

    for index, text in enumerate(texts):
        label = text.strip().lower().rstrip(":")

        if label == "seller":
            # OCR reads the two-column Seller/Client labels before their values.
            if index + 1 < len(texts) and texts[index + 1].strip().lower().rstrip(":") == "client":
                return texts[index + 2].strip() if index + 2 < len(texts) else None
            if index + 1 < len(texts):
                value = texts[index + 1].strip()
                return value or None

        if label == "bill from":
            # The first company after the Bill from/Bill To labels is the supplier.
            value_index = index + 1
            if value_index < len(texts) and texts[value_index].strip().lower().rstrip(":") == "bill to":
                value_index += 1
            if value_index < len(texts):
                return texts[value_index].strip() or None

    # A Bill To label alone identifies the customer, not the supplier.
    return None


def _get_amount(text):
    # Keep the extracted amount as a string, using a decimal point for decimal commas.
    match = re.search(r"\d[\d,]*(?:\.\d{1,2})?", text.replace(" ", ""))
    if not match:
        return None

    amount = match.group()
    if "," in amount and "." not in amount:
        amount = amount.replace(",", ".")
    elif "," in amount and "." in amount:
        amount = amount.replace(",", "")

    return amount


def _normalize_invoice_date(value):
    # Store complete dates as ISO dates so every invoice uses the same structure.
    formats = (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(value.strip(), date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _box_center(detection):
    if len(detection) < 2 or detection[0] is None or len(detection[0]) == 0:
        return None
    points = detection[0]
    if any(len(point) < 2 for point in points):
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(xs), max(xs)


def _get_day_near_date_label(result, label_index):
    # Recover a day EasyOCR may detect as a separate, misplaced text box.
    label_center = _box_center(result[label_index])
    if label_center is None:
        return None
    label_x, label_y, _, _ = label_center
    year_matches = []
    for detection in result:
        if len(detection) < 2:
            continue
        year_match = re.fullmatch(r"\s*((?:19|20)\d{2})\s*", detection[1])
        if not year_match:
            continue
        year_center = _box_center(detection)
        if year_center is None:
            continue
        year_x, year_y, year_left, _ = year_center
        if year_x > label_x and abs(year_y - label_y) <= 36:
            year_matches.append((abs(year_y - label_y), year_x, year_left))

    if not year_matches:
        return None
    _, year_x, year_left = min(year_matches)

    day_candidates = []
    for detection in result:
        if len(detection) < 2:
            continue
        day_match = re.fullmatch(r"\s*(\d{1,2})[,./]?\s*", detection[1])
        if not day_match:
            continue
        day = int(day_match.group(1))
        if not 1 <= day <= 31:
            continue

        day_center = _box_center(detection)
        if day_center is None:
            continue
        day_x, day_y, _, _ = day_center
        if label_x < day_x < year_x and abs(day_y - label_y) <= 26:
            day_candidates.append((abs(year_left - day_x), day))

    return min(day_candidates)[1] if day_candidates else None


def extract_date_facture(result):
    texts = _get_texts(result)

    for index, text in enumerate(texts):
        # These labels cover the three invoice layouts in the project.
        if re.search(r"\b(date of issue|invoice date|date)\s*:", text, re.IGNORECASE):
            inline_date = re.split(r":", text, maxsplit=1)[1].strip(" :-")
            date = inline_date
            if not date and index + 1 < len(texts):
                date = texts[index + 1].strip(" :-")

            # Some layouts OCR the invoice date's year on a nearby separate line.
            if date and not re.search(r"\b(?:19|20)\d{2}\b", date):
                year_start = index + 2 if not inline_date else index + 1
                for nearby_text in texts[year_start:year_start + 4]:
                    if re.search(r"\bdue date\b", nearby_text, re.IGNORECASE):
                        break
                    year_match = re.fullmatch(r"\s*((?:19|20)\d{2})\s*", nearby_text)
                    if year_match:
                        date += " " + year_match.group(1)
                        break

            normalized = _normalize_invoice_date(date)
            if normalized:
                return normalized

            # The day can be spatially between the month label and year even
            # when EasyOCR returns it as a separate item at the end of its list.
            day = _get_day_near_date_label(result, index)
            month_year = re.fullmatch(
                r"([A-Za-z]+)\s+((?:19|20)\d{2})",
                date.strip().replace(",", ""),
            )
            if day and month_year:
                normalized = _normalize_invoice_date(
                    f"{month_year.group(1)} {day}, {month_year.group(2)}"
                )
                if normalized:
                    return normalized
            return None

    return None


def extract_montant_ht(result):
    texts = _get_texts(result)

    # Layouts 2 and 3 show the net amount after "Subtotal".
    for index, text in enumerate(texts):
        if "subtotal" in text.lower():
            amount = _get_amount(text)
            if amount:
                return amount

            for following_text in texts[index + 1:index + 3]:
                amount = _get_amount(following_text)
                if amount:
                    return amount

    # Layout 1 shows the net amount in the summary's "Total" row.
    total_indexes = [
        index for index, text in enumerate(texts)
        if text.strip().lower() == "total"
    ]
    if total_indexes:
        for following_text in texts[total_indexes[-1] + 1:total_indexes[-1] + 4]:
            amount = _get_amount(following_text)
            if amount:
                return amount

    return None


def extract_montant_ttc(result):
    texts = _get_texts(result)

    # Find "Total Due" whether OCR returned it on one line or two.
    for index, text in enumerate(texts):
        label = text.strip().lower()
        is_total_due = "total due" in label or (
            label == "total"
            and index + 1 < len(texts)
            and texts[index + 1].strip().lower() == "due"
        )

        if is_total_due:
            # The amount may follow the label on the same line or the next line.
            amount = _get_amount(text)
            if amount:
                return amount

            start = index + 2 if label == "total" else index + 1
            for following_text in texts[start:start + 3]:
                amount = _get_amount(following_text)
                if amount:
                    return amount

    # Layout 1 labels the summary row "Total"; its last amount is the gross total.
    total_indexes = [
        index for index, text in enumerate(texts)
        if text.strip().lower() == "total"
    ]
    if total_indexes:
        amounts = [
            amount
            for text in texts[total_indexes[-1] + 1:total_indexes[-1] + 4]
            if (amount := _get_amount(text))
        ]
        if amounts:
            return amounts[-1]

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        import easyocr

        image_path = Path(sys.argv[2])
        reader = easyocr.Reader(["en"])
        detections = reader.readtext(str(image_path), detail=1)
        fields = {
            "numeroFacture": extract_invoice_number(detections),
            "fournisseur": extract_supplier(detections),
            "dateFacture": extract_date_facture(detections),
            "montantHT": extract_montant_ht(detections),
            "montantTTC": extract_montant_ttc(detections),
        }
        print("\t".join(str(value) if value is not None else "" for value in fields.values()))
        sys.exit(0)

    # Pass an OCR text file path to test another invoice; defaults to a project sample.
    ocr_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "ocr_results/model_3/invoice_001.txt"
    )

    # Convert the saved OCR lines into the tuple shape expected by the extractors.
    with ocr_file.open(encoding="utf-8") as file:
        result = [(None, line.strip(), 1.0) for line in file if line.strip()]

    print(f"Invoice number: {extract_invoice_number(result)}")
    print(f"Fournisseur:    {extract_supplier(result)}")
    print(f"Date facture:  {extract_date_facture(result)}")
    print(f"Montant HT:    {extract_montant_ht(result)}")
    print(f"Montant TTC:   {extract_montant_ttc(result)}")
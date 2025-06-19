import json
import os


def parse_clinical_trials_json_folder(input_folder):
    all_texts = []
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".json"):
            path = os.path.join(input_folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = flatten_clinical_trials_json(data)
            all_texts.append((fname, text))
    return all_texts


def flatten_clinical_trials_json(data):
    parts = []
    protocol = data.get("protocolSection", {})
    id_mod = protocol.get("identificationModule", {})
    brief_title = id_mod.get("briefTitle", "")
    official_title = id_mod.get("officialTitle", "")
    parts.append(brief_title)
    parts.append(official_title)

    desc_mod = protocol.get("descriptionModule", {})
    brief_summary = desc_mod.get("briefSummary", "")
    parts.append(brief_summary)

    outcomes_mod = protocol.get("outcomesModule", {})
    primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
    for outcome in primary_outcomes:
        parts.append(outcome.get("measure", ""))
        parts.append(outcome.get("description", ""))

    return "\n\n".join([p for p in parts if p])

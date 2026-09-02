"""Apply the documented manual correctness review to the fixed audit sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import mcq_content_sha256, read_jsonl, write_jsonl


INCORRECT = {
    "self-56bd76eacc4bc96e": (
        "INR 4.2 with non-major gingival bleeding does not generally justify immediate IV vitamin K; "
        "holding warfarin and risk-guided follow-up is missing from the choices."
    ),
    "evol-236ebd633eb70f4f": (
        "A prior non-severe penicillin rash does not make every cephalosporin contraindicated; the recurrence/new-rash "
        "timeline is also insufficient to establish azithromycin as the unique best answer."
    ),
    "self-9011b03de3ccfdc4": (
        "Mifepristone is not a universal first-line treatment for symptomatic fibroids; anemia, fertility goals, "
        "contraindications, and standard bleeding-control options are not specified."
    ),
    "self-24f48c77d591c866": (
        "Diffuse nodules, ground glass, and restriction without hilar adenopathy or a perilymphatic distribution do not "
        "uniquely identify sarcoidosis over hypersensitivity pneumonitis, miliary infection, or other ILD."
    ),
    "self-4fba2a21c0cdd50c": (
        "INR 4.2 with minor mucocutaneous bleeding makes immediate IV vitamin K overly aggressive; no appropriate "
        "hold-warfarin option is offered."
    ),
}

REVIEW_RESULTS = {
    "evol-ddfbab6412587730": True,
    "self-56bd76eacc4bc96e": False,
    "self-73bdf56da2b8bcab": True,
    "self-5dbac275d46fd629": True,
    "self-713210d39271b397": True,
    "evol-236ebd633eb70f4f": False,
    "self-4c268ada682b4622": True,
    "self-69dbe13e7d8cc935": True,
    "evol-fbfc167ee935906b": True,
    "evol-57d99ad393b61427": True,
    "self-9011b03de3ccfdc4": False,
    "self-5c70f7d7eb4fe2bf": True,
    "self-0eb1151c7f12f1aa": True,
    "self-6487340962f3611b": True,
    "self-ef915dec924dd815": True,
    "evol-15666c85d1b51986": True,
    "evol-3999c404b6df48d9": True,
    "evol-c841ac195fa299c3": True,
    "self-24f48c77d591c866": False,
    "self-cfbd3f7ff2b2fba5": True,
    "evol-000d5b258afbce2a": True,
    "evol-0f9410dabd1573a3": True,
    "self-2c0326461c4594ba": True,
    "self-ecf3f1e432306efd": True,
    "self-474054584a61f443": True,
    "evol-cd81da1fd1db08f8": True,
    "self-4fba2a21c0cdd50c": False,
    "self-59affd5ce1212f19": True,
    "self-8849d999dfdf7091": True,
    "self-0f51faa7a3eec018": True,
}

REVIEW_HASHES = {
    "evol-ddfbab6412587730": "724e79c711c6b6b8b010624955e602b814afef6327a729f8a923ac4a7d0f9c3e",
    "self-56bd76eacc4bc96e": "7c48289244a31e825622c47730766fcc803b444e807314b1ef413dcfa75106f5",
    "self-73bdf56da2b8bcab": "5a42b2453568687aad5751734c545288c4215efc08dc506905a26ce81d5458ad",
    "self-5dbac275d46fd629": "96f7d78906b2e2481ae5798e71a37e12e3b47550bfa93c20b8ffab41e4901dee",
    "self-713210d39271b397": "369686c420c543edea02e6dc4759827e343982336667273b376b6929a2bc99e7",
    "evol-236ebd633eb70f4f": "f8857d4d10cfe0115021516f0781e5eb1887387821fbe502fc2023e4589d1072",
    "self-4c268ada682b4622": "d632209f4c9a298f3276fccf295aa6619f8f1227909b59f36de63e999f8c04c2",
    "self-69dbe13e7d8cc935": "8f141004f8b0dfc1cd2ee18c00b527a9c50d4f804ca01ea572e25e82a0935412",
    "evol-fbfc167ee935906b": "f120b4e5abe53f0e1cdd274e256fe915c297ee2352cc263139c50f78c4157f54",
    "evol-57d99ad393b61427": "87755775d3e5680d7e65e784967235d09beeca978db3ef2de720fab2966a2ea8",
    "self-9011b03de3ccfdc4": "7b2c4e97ebe48e52ccd5ced57866c8ebd7e7116d051c465924d07f315074a68a",
    "self-5c70f7d7eb4fe2bf": "58578c139193d22c3a73e4c52f581dfbc8be965a7211d05823ba00dfb4356343",
    "self-0eb1151c7f12f1aa": "21fe08236b8b414df01c8606b996d953d574199e3a1bae9c0e039459d9493d76",
    "self-6487340962f3611b": "4d1d973084673631f84830b8923f895c53385afba25508aba951a5e582b4296c",
    "self-ef915dec924dd815": "375affc061e877e3d50d406808934be6eb0a424d2d6e3af433927cb928c77737",
    "evol-15666c85d1b51986": "ecdf405c3b1a38d19894cdfabd22ae5847ccc40932d3d76f6499eaf51efc9ed8",
    "evol-3999c404b6df48d9": "b0811ffb9abfd28c3402954334e83799695adb487592c06ef59210dd528faa07",
    "evol-c841ac195fa299c3": "eba3853caa3e288b28fd250bb04940f47844935ecad1748ed3d547126ebc483e",
    "self-24f48c77d591c866": "6b585483d2866cffbd5decaf3b20c33d32d29b37966e2336449b012c9b869dc0",
    "self-cfbd3f7ff2b2fba5": "4c8f6bbc43bda656f9b4f08bc5c4f33f9c9e30ebd052c2149d1a1863973d7676",
    "evol-000d5b258afbce2a": "46043ba9d72630a2ebe0a3ed5170cc30aac624d4e541064566c37e9595407170",
    "evol-0f9410dabd1573a3": "c31e971f210e25a64ad3635bccff335ec7105fd19cd03f827e56c253b81d3135",
    "self-2c0326461c4594ba": "1102402c8d7114489b2afec73f17b720eed262d386803b2ca1366e0d87d72616",
    "self-ecf3f1e432306efd": "3e082958a0ca5e5e81fe382819a580ee76c7030562233341220b569ed5c224ec",
    "self-474054584a61f443": "abe26b65e33650abf3a808727da4f32a839988fb7e65b6b81e9b9dd27dafc212",
    "evol-cd81da1fd1db08f8": "7e63985280c8d259e412f755528d3b70c94e3e81b19d8c806065bee65f5ad14e",
    "self-4fba2a21c0cdd50c": "b44b1e6c06df79f2e9abc6f6e40afd56fece810582b6447e90dc373793d07b97",
    "self-59affd5ce1212f19": "0499377b82dd5a2f8504a1e55e1b6209ed41a393316a87ff55f2f007f53d1706",
    "self-8849d999dfdf7091": "a80a5db3f2b24b6e4355ffcbc9bc3a1c32214fa8588ebfd2a0c45d3c4a696178",
    "self-0f51faa7a3eec018": "d9e1f3d50661eb6e1c5a226ef3ab0627cc751534cf455363f3c3132ef01fe7fd",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.audit)
    record_ids = [row.get("synthetic_id") for row in rows]
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(REVIEW_RESULTS):
        missing = sorted(set(REVIEW_RESULTS) - set(record_ids))
        unexpected = sorted(set(record_ids) - set(REVIEW_RESULTS))
        raise SystemExit(f"audit sample mismatch; missing={missing}, unexpected={unexpected}")
    for row in rows:
        record_id = row.get("synthetic_id")
        content_sha256 = mcq_content_sha256(row)
        if content_sha256 != REVIEW_HASHES[record_id]:
            raise SystemExit(f"reviewed content changed for {record_id}; refuse to reuse label")
        row.pop("human_correct", None)
        row.pop("human_notes", None)
        row.pop("manual_ai_correct", None)
        row.pop("manual_ai_notes", None)
        row["content_sha256"] = content_sha256
        row["review_correct"] = REVIEW_RESULTS[record_id]
        row["review_notes"] = INCORRECT.get(
            record_id, "Answer and explanation appear medically coherent for the stated question."
        )
        row["reviewer"] = "AI review (not clinician/human), 2026-08-24"
        row["reviewer_kind"] = "ai_nonclinician"
    write_jsonl(args.audit, rows)
    correct = sum(row["review_correct"] for row in rows)
    if args.metrics:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        metrics.pop("human_audit", None)
        metrics.pop("manual_ai_review", None)
        metrics["manual_review"] = {
            "path": str(args.audit), "sample_size": len(rows), "reviewed": len(rows),
            "accuracy": round(correct / len(rows), 6),
            "reviewer_kinds": ["ai_nonclinician"],
        }
        args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[audit] reviewed={len(rows)} correct={correct} accuracy={correct / max(len(rows), 1):.3f}")


if __name__ == "__main__":
    main()

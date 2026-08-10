import argparse
import json
import re
from pathlib import Path

import SimpleITK as sitk
from radiomics import featureextractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "NPC_pre" / "T1" / "image"
LABEL_DIR = PROJECT_ROOT / "data" / "NPC_pre" / "T1" / "label"
OUTPUT_DIR = PROJECT_ROOT / "data" / "radiomics"

CASE_ID_RE = re.compile(r"(\d{3})")


def case_id_from_filename(path: Path) -> str:
    match = CASE_ID_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not find a 3-digit case id in {path.name}")
    return match.group(1)


def extract_case(extractor, image_path: Path, mask_path: Path) -> dict:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    result = extractor.execute(image, mask)
    features = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        features[key] = float(value) if hasattr(value, "item") else value
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing JSON files",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = featureextractor.RadiomicsFeatureExtractor()

    image_paths = sorted(IMAGE_DIR.glob("*.nii.gz"))
    for image_path in image_paths:
        case_id = case_id_from_filename(image_path)
        mask_path = LABEL_DIR / image_path.name
        if not mask_path.exists():
            print(f"[skip] Case{case_id}: no matching label at {mask_path}")
            continue

        out_path = OUTPUT_DIR / f"Case{case_id}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] Case{case_id}: {out_path.name} already exists")
            continue

        print(f"[run ] Case{case_id}")
        features = extract_case(extractor, image_path, mask_path)
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)


if __name__ == "__main__":
    main()

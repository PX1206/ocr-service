"""Portrait (head + shoulders) validation orchestration."""

from __future__ import annotations

from typing import Any, Dict

from app.common.portrait_rules import validate_portrait
from app.common.pose_engine import run_pose


def validate_portrait_image(image_path: str) -> Dict[str, Any]:
    pose_result = run_pose(image_path)
    person_count = int(pose_result.get("person_count") or 0)
    persons = pose_result.get("persons") or []

    keypoints = None
    mean_score = None
    if person_count == 1 and persons:
        keypoints = persons[0].get("keypoints")
        mean_score = persons[0].get("mean_score")

    validation = validate_portrait(
        image_width=int(pose_result.get("image_width") or 0),
        image_height=int(pose_result.get("image_height") or 0),
        person_count=person_count,
        keypoints=keypoints,
        keypoint_scores=[mean_score] if mean_score is not None else None,
    )
    return validation.to_dict()


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.parsers.portrait_validator <image_path>")
        raise SystemExit(1)

    print(json.dumps(validate_portrait_image(sys.argv[1]), ensure_ascii=False, indent=2))

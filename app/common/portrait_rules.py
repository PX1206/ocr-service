"""Business rules for head-and-shoulders portrait validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# COCO 17 keypoints
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

REQUIRED_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER)

REASON_MESSAGES = {
    "NO_PERSON": "未检测到人物，请对准引导框拍摄",
    "MULTIPLE_PERSONS": "画面中检测到多人，请确保仅您一人在镜头内",
    "FACE_NOT_VISIBLE": "未清晰检测到面部，请正面对准引导框",
    "SHOULDER_NOT_VISIBLE": "未清晰检测到双肩，请后退一些将肩膀纳入画面",
    "SHOULDER_CROPPED": "请将双肩完整纳入画面",
    "FACE_TOO_SMALL": "请靠近一些，确保人脸清晰可见",
    "PERSON_TOO_SMALL": "人物过小，请靠近镜头后重拍",
    "PASSED": "构图合格",
}


@dataclass
class PortraitValidationResult:
    passed: bool
    reason: Optional[str]
    message: str
    person_count: int = 0
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "passed": self.passed,
            "reason": self.reason,
            "message": self.message,
            "person_count": self.person_count,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


def _point_score(keypoints: Sequence[Sequence[float]], index: int) -> float:
    return float(keypoints[index][2])


def _point_xy(keypoints: Sequence[Sequence[float]], index: int) -> Tuple[float, float]:
    return float(keypoints[index][0]), float(keypoints[index][1])


def _within_margin(x: float, y: float, width: int, height: int, margin_ratio: float) -> bool:
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return margin_x <= x <= width - margin_x and margin_y <= y <= height - margin_y


def validate_portrait(
    *,
    image_width: int,
    image_height: int,
    person_count: int,
    keypoints: Optional[Sequence[Sequence[float]]],
    keypoint_scores: Optional[Sequence[float]] = None,
    score_threshold: float = 0.5,
    edge_margin_ratio: float = 0.02,
    min_face_height_ratio: float = 0.12,
    min_person_height_ratio: float = 0.25,
) -> PortraitValidationResult:
    """Validate a single-person head-and-shoulders portrait."""
    if person_count <= 0:
        return PortraitValidationResult(
            passed=False,
            reason="NO_PERSON",
            message=REASON_MESSAGES["NO_PERSON"],
            person_count=0,
        )

    if person_count > 1:
        return PortraitValidationResult(
            passed=False,
            reason="MULTIPLE_PERSONS",
            message=REASON_MESSAGES["MULTIPLE_PERSONS"],
            person_count=person_count,
        )

    if not keypoints or len(keypoints) < 17:
        return PortraitValidationResult(
            passed=False,
            reason="FACE_NOT_VISIBLE",
            message=REASON_MESSAGES["FACE_NOT_VISIBLE"],
            person_count=1,
        )

    low_points: List[int] = []
    for index in REQUIRED_KEYPOINTS:
        score = _point_score(keypoints, index)
        if score < score_threshold:
            low_points.append(index)

    if any(index in low_points for index in (NOSE, LEFT_EYE, RIGHT_EYE)):
        return PortraitValidationResult(
            passed=False,
            reason="FACE_NOT_VISIBLE",
            message=REASON_MESSAGES["FACE_NOT_VISIBLE"],
            person_count=1,
            details={"low_score_keypoints": low_points},
        )

    if LEFT_SHOULDER in low_points or RIGHT_SHOULDER in low_points:
        return PortraitValidationResult(
            passed=False,
            reason="SHOULDER_NOT_VISIBLE",
            message=REASON_MESSAGES["SHOULDER_NOT_VISIBLE"],
            person_count=1,
            details={"low_score_keypoints": low_points},
        )

    for index in (LEFT_SHOULDER, RIGHT_SHOULDER):
        x, y = _point_xy(keypoints, index)
        if not _within_margin(x, y, image_width, image_height, edge_margin_ratio):
            return PortraitValidationResult(
                passed=False,
                reason="SHOULDER_CROPPED",
                message=REASON_MESSAGES["SHOULDER_CROPPED"],
                person_count=1,
            )

    nose_y = _point_xy(keypoints, NOSE)[1]
    left_shoulder_y = _point_xy(keypoints, LEFT_SHOULDER)[1]
    right_shoulder_y = _point_xy(keypoints, RIGHT_SHOULDER)[1]
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0
    face_height = max(shoulder_y - nose_y, 1.0)
    face_height_ratio = face_height / max(image_height, 1)

    if face_height_ratio < min_face_height_ratio:
        return PortraitValidationResult(
            passed=False,
            reason="FACE_TOO_SMALL",
            message=REASON_MESSAGES["FACE_TOO_SMALL"],
            person_count=1,
            details={"face_height_ratio": round(face_height_ratio, 4)},
        )

    ys = [_point_xy(keypoints, index)[1] for index in REQUIRED_KEYPOINTS]
    person_height_ratio = (max(ys) - min(ys)) / max(image_height, 1)
    if person_height_ratio < min_person_height_ratio:
        return PortraitValidationResult(
            passed=False,
            reason="PERSON_TOO_SMALL",
            message=REASON_MESSAGES["PERSON_TOO_SMALL"],
            person_count=1,
            details={"person_height_ratio": round(person_height_ratio, 4)},
        )

    details: Dict[str, Any] = {
        "face_height_ratio": round(face_height_ratio, 4),
        "person_height_ratio": round(person_height_ratio, 4),
    }
    if keypoint_scores:
        details["mean_keypoint_score"] = round(float(keypoint_scores[0]), 4)

    return PortraitValidationResult(
        passed=True,
        reason=None,
        message=REASON_MESSAGES["PASSED"],
        person_count=1,
        details=details,
    )

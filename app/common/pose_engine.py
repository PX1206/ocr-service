"""Lazy-loaded PicoDet + PP-TinyPose inference for portrait validation."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.common.logging_config import get_logger

logger = get_logger("app.pose")

DEFAULT_DET_MODEL_DIR = "/app/models/picodet_s_320_pedestrian"
DEFAULT_KPT_MODEL_DIR = "/app/models/tinypose_256x192"
DEFAULT_DEPLOY_DIR = "/opt/PaddleDetection/deploy/python"

_engine_lock = threading.Lock()
_engine: Optional["_PoseEngine"] = None


def _resolve_deploy_dir() -> str:
    deploy_dir = os.getenv("PADDLE_DET_DEPLOY_DIR", DEFAULT_DEPLOY_DIR).strip()
    if not os.path.isdir(deploy_dir):
        raise RuntimeError(
            f"PaddleDetection deploy dir not found: {deploy_dir}. "
            "Set PADDLE_DET_DEPLOY_DIR or build image with PaddleDetection clone."
        )
    return deploy_dir


def _ensure_deploy_path() -> str:
    deploy_dir = _resolve_deploy_dir()
    if deploy_dir not in sys.path:
        sys.path.insert(0, deploy_dir)
    return deploy_dir


def _load_deploy_module(deploy_dir: str, module_name: str):
    """Load deploy/python/*.py explicitly to avoid wrong picodet_postprocess shadowing."""
    import importlib.util

    module_path = os.path.join(deploy_dir, f"{module_name}.py")
    if not os.path.isfile(module_path):
        raise RuntimeError(f"Missing PaddleDetection deploy module: {module_path}")

    spec = importlib.util.spec_from_file_location(f"pd_deploy_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load deploy module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[f"pd_deploy_{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def _patch_picodet_detector_postprocess(deploy_dir: str) -> None:
    """Fix PicoDetPostProcess strides arg mismatch across PaddleDetection deploy versions."""
    import infer as pd_infer

    if getattr(pd_infer.DetectorPicoDet, "_nh_portrait_postprocess_patched", False):
        return

    pp_module = _load_deploy_module(deploy_dir, "picodet_postprocess")

    def postprocess(self, inputs, result):
        np_score_list = result["boxes"]
        np_boxes_list = result["boxes_num"]
        input_shape = inputs["image"].shape[2:]
        ori_shape = inputs["im_shape"]
        scale_factor = inputs["scale_factor"]
        fpn_stride = self.pred_config.fpn_stride
        nms_threshold = self.pred_config.nms["nms_threshold"]

        last_error = None
        for factory in (
            lambda: pp_module.PicoDetPostProcess(
                input_shape,
                ori_shape,
                scale_factor,
                strides=fpn_stride,
                nms_threshold=nms_threshold,
            ),
            lambda: pp_module.PicoDetPostProcess(
                input_shape,
                ori_shape,
                fpn_stride,
                scale_factor,
                nms_threshold=nms_threshold,
            ),
        ):
            try:
                postprocessor = factory()
                break
            except TypeError as ex:
                last_error = ex
                postprocessor = None
        else:
            raise last_error or RuntimeError("PicoDetPostProcess init failed")

        np_boxes, np_boxes_num = postprocessor(np_score_list, np_boxes_list)
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)

    pd_infer.DetectorPicoDet.postprocess = postprocess
    pd_infer.DetectorPicoDet._nh_portrait_postprocess_patched = True
    logger.info("patched DetectorPicoDet.postprocess for PicoDet stride compatibility")


class _PoseEngine:
    def __init__(
        self,
        det_model_dir: str,
        kpt_model_dir: str,
        device: str = "CPU",
        det_threshold: float = 0.5,
        cpu_threads: int = 4,
    ) -> None:
        deploy_dir = _ensure_deploy_path()
        _patch_picodet_detector_postprocess(deploy_dir)
        from infer import DetectorPicoDet
        from keypoint_infer import KeyPointDetector
        from keypoint_postprocess import translate_to_ori_images
        from preprocess import decode_image

        self._decode_image = decode_image
        self._translate_to_ori_images = translate_to_ori_images
        self._det_threshold = det_threshold

        logger.info(
            "loading portrait models det=%s kpt=%s device=%s",
            det_model_dir,
            kpt_model_dir,
            device,
        )
        started = time.perf_counter()
        self._detector = DetectorPicoDet(
            det_model_dir,
            device=device,
            run_mode="paddle",
            cpu_threads=cpu_threads,
            enable_mkldnn=False,
            threshold=det_threshold,
        )
        self._keypoint_detector = KeyPointDetector(
            kpt_model_dir,
            device=device,
            run_mode="paddle",
            batch_size=1,
            cpu_threads=cpu_threads,
            enable_mkldnn=False,
            use_dark=True,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("portrait models ready elapsed_ms=%d", elapsed_ms)

    def analyze(self, image_path: str) -> Dict[str, Any]:
        image, _ = self._decode_image(image_path, {})
        image_height, image_width = image.shape[:2]

        det_results = self._detector.predict_image([image], visual=False)
        det_results = self._detector.filter_box(det_results, self._det_threshold)
        person_count = int(det_results["boxes_num"][0]) if len(det_results["boxes_num"]) else 0

        if person_count <= 0:
            return {
                "image_width": image_width,
                "image_height": image_height,
                "person_count": 0,
                "persons": [],
            }

        rec_images, records, det_rects = self._keypoint_detector.get_person_from_rect(
            image, det_results
        )
        if not det_rects:
            return {
                "image_width": image_width,
                "image_height": image_height,
                "person_count": person_count,
                "persons": [],
            }

        keypoint_results = self._keypoint_detector.predict_image(
            rec_images, run_benchmark=False, repeats=1, visual=False
        )
        keypoint_vectors, score_vectors = self._translate_to_ori_images(
            keypoint_results, np.array(records)
        )

        persons: List[Dict[str, Any]] = []
        for index, rect in enumerate(det_rects):
            if index >= len(keypoint_vectors):
                break
            keypoints = keypoint_vectors[index].tolist()
            scores = score_vectors[index].tolist() if index < len(score_vectors) else []
            persons.append(
                {
                    "bbox": [float(v) for v in rect],
                    "keypoints": keypoints,
                    "mean_score": float(np.mean(scores)) if scores else 0.0,
                }
            )

        return {
            "image_width": image_width,
            "image_height": image_height,
            "person_count": person_count,
            "persons": persons,
        }


def get_pose_engine() -> _PoseEngine:
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        det_model_dir = os.getenv("PORTRAIT_DET_MODEL_DIR", DEFAULT_DET_MODEL_DIR).strip()
        kpt_model_dir = os.getenv("PORTRAIT_KPT_MODEL_DIR", DEFAULT_KPT_MODEL_DIR).strip()
        device = os.getenv("PORTRAIT_DEVICE", "CPU").strip().upper()
        det_threshold = float(os.getenv("PORTRAIT_DET_THRESHOLD", "0.5"))
        cpu_threads = int(os.getenv("PORTRAIT_CPU_THREADS", "4"))

        for path, label in (
            (det_model_dir, "det"),
            (kpt_model_dir, "kpt"),
        ):
            if not os.path.isdir(path):
                raise RuntimeError(f"Portrait {label} model dir not found: {path}")

        _engine = _PoseEngine(
            det_model_dir=det_model_dir,
            kpt_model_dir=kpt_model_dir,
            device=device,
            det_threshold=det_threshold,
            cpu_threads=cpu_threads,
        )
        return _engine


def warmup_pose_engine() -> None:
    if os.getenv("PORTRAIT_WARMUP", "false").lower() not in {"1", "true", "yes"}:
        return
    get_pose_engine()
    logger.info("portrait pose engine warmup complete")


def run_pose(image_path: str) -> Dict[str, Any]:
    return get_pose_engine().analyze(image_path)

#!/usr/bin/env python3
"""
Lissajous Pipeline - 통합 이미지 처리 파이프라인

5단계 파이프라인:
1. HEIF 이미지 로드
2. 디스플레이 영역 검출
3. 곡선 추출 & 골격화
4. 레이아웃 보정
5. 지표 계산

사용법:
    python lissajous_pipeline.py image.HIF -fx 1350 -fy 1480 -phase 0
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# HEIF 지원
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image
from skimage import morphology
import matplotlib.pyplot as plt
from math import gcd, sqrt, log, pi


@dataclass
class LissajousMetrics:
    """리사쥬 패턴 분석 지표 (C++ lissajous analyzer와 동일)"""
    # 4가지 핵심 지표
    fill_factor: float      # FF: 채워진 픽셀 비율 (0~1)
    cv: float               # CV: Coefficient of Variation (균일도)
    entropy: float          # H: Shannon Entropy
    max_h: float            # max_h: 최대 공백 크기
    
    # 픽셀 정보
    total_pixels: int       # 전체 그리드 픽셀 수
    filled_pixels: int      # 방문한 픽셀 수
    
    # 그리드 정보
    grid_size: Tuple[int, int]  # (width, height)
    
    def __str__(self) -> str:
        return (f"FF={self.fill_factor*100:.2f}%, CV={self.cv:.4f}, "
                f"H={self.entropy:.4f}, max_h={self.max_h:.6f}")


class MetricsCalculator:
    """
    그리드 기반 리사쥬 지표 계산기
    C++ metrics.cpp 로직을 Python으로 포팅
    """
    
    def __init__(self, grid_size: int = 256, fov_x: float = 10.0, fov_y: float = 10.0):
        """
        Args:
            grid_size: 그리드 해상도 (256x256 기본)
            fov_x, fov_y: FOV 범위 (도 단위)
        """
        self.grid_size_x = grid_size
        self.grid_size_y = grid_size
        self.fov_x = fov_x
        self.fov_y = fov_y
    
    def generate_trajectory(self, fx: int, fy: int, phase_deg: float, 
                            duration: float = None) -> np.ndarray:
        """
        리사쥬 궤적 생성
        
        Args:
            fx, fy: 주파수 (Hz)
            phase_deg: 위상차 (도)
            duration: 시간 (초), None이면 한 주기
        
        Returns:
            np.ndarray: (N, 2) 형태의 궤적 좌표 (각도 단위)
        """
        g = gcd(fx, fy)
        nx = fx // g
        ny = fy // g
        f0 = g
        
        # 한 주기 또는 지정된 시간
        if duration is None:
            pattern_period = 1.0 / f0
            duration = pattern_period
        
        # 샘플링 레이트: 논문 기반 2*nx*ny*f0
        sampling_rate = 2 * nx * ny * f0
        dt = 1.0 / sampling_rate
        steps = int(duration / dt)
        
        phase = np.radians(phase_deg)
        t = np.arange(steps) * dt
        
        # 리사쥬 방정식 (FOV 범위 내 각도)
        x = (self.fov_x / 2.0) * np.sin(2 * pi * fx * t + phase)
        y = (self.fov_y / 2.0) * np.sin(2 * pi * fy * t)
        
        return np.column_stack((x, y))
    
    def calculate_stats(self, trajectory: np.ndarray) -> LissajousMetrics:
        """
        궤적에서 4가지 지표 계산 (C++ calculateStats 포팅)
        
        Args:
            trajectory: (N, 2) 형태의 좌표 배열
        
        Returns:
            LissajousMetrics: FF, CV, Entropy, max_h
        """
        total_pixels = self.grid_size_x * self.grid_size_y
        visited = np.zeros(total_pixels, dtype=bool)
        hits = np.zeros(total_pixels, dtype=np.uint32)
        
        # 궤적을 그리드에 매핑
        for p in trajectory:
            idx = self._to_cell(p[0], p[1])
            if not visited[idx]:
                visited[idx] = True
            hits[idx] += 1
        
        filled_pixels = visited.sum()
        
        # 1. Fill Factor (FF)
        fill_factor = filled_pixels / total_pixels
        
        # 2. CV (Coefficient of Variation) - 방문한 픽셀들만
        if filled_pixels > 0:
            visited_hits = hits[visited]
            mu = visited_hits.mean()
            sigma = visited_hits.std()
            cv = sigma / mu if mu > 0 else 0
        else:
            cv = 0
        
        # 3. Shannon Entropy
        total_hits = hits.sum()
        entropy = 0.0
        if total_hits > 0:
            for h in hits:
                if h > 0:
                    p = h / total_hits
                    entropy -= p * log(p)
        
        # 4. max_h는 별도 계산 (주파수 필요)
        max_h = 0.0
        
        return LissajousMetrics(
            fill_factor=fill_factor,
            cv=cv,
            entropy=entropy,
            max_h=max_h,
            total_pixels=total_pixels,
            filled_pixels=int(filled_pixels),
            grid_size=(self.grid_size_x, self.grid_size_y)
        )
    
    def calculate_max_h(self, fx: int, fy: int) -> float:
        """
        max_h 계산 (C++ approximateMaxH 포팅)
        리사쥬 궤적 사이의 최대 간격
        
        Args:
            fx, fy: 주파수 (Hz)
        
        Returns:
            float: max_h 값 (FOV 단위)
        """
        g = gcd(fx, fy)
        nx = fx // g
        ny = fy // g
        
        if nx % 2 == 0 or ny % 2 == 0:
            # 하나라도 짝수인 경우
            edge_x = 2.0 / nx
            edge_y = 2.0 / ny
        else:
            # 둘 다 홀수인 경우
            edge_x = 1.0 / nx
            edge_y = 1.0 / ny
        
        max_h_normalized = sqrt(edge_x**2 + edge_y**2) / 2.0
        
        # FOV 단위로 변환
        return max_h_normalized * self.fov_y
    
    def calculate_full_stats(self, fx: int, fy: int, phase_deg: float,
                              duration: float = None) -> LissajousMetrics:
        """
        전체 지표 계산 (궤적 생성 + 통계 + max_h)
        
        Args:
            fx, fy: 주파수 (Hz)
            phase_deg: 위상차 (도)
            duration: 시간 (초)
        
        Returns:
            LissajousMetrics: 모든 지표 포함
        """
        trajectory = self.generate_trajectory(fx, fy, phase_deg, duration)
        stats = self.calculate_stats(trajectory)
        stats.max_h = self.calculate_max_h(fx, fy)
        return stats
    
    def _to_cell(self, x: float, y: float) -> int:
        """각도 좌표를 그리드 셀 인덱스로 변환"""
        ix = int((x + self.fov_x / 2.0) / self.fov_x * self.grid_size_x)
        iy = int((y + self.fov_y / 2.0) / self.fov_y * self.grid_size_y)
        
        # 경계 처리
        ix = max(0, min(self.grid_size_x - 1, ix))
        iy = max(0, min(self.grid_size_y - 1, iy))
        
        return iy * self.grid_size_x + ix
    
    def get_grid_image(self, trajectory: np.ndarray) -> np.ndarray:
        """
        궤적을 그리드 이미지로 변환 (시각화용)
        
        Returns:
            np.ndarray: (grid_size_y, grid_size_x) 형태의 방문 횟수 맵
        """
        grid = np.zeros((self.grid_size_y, self.grid_size_x), dtype=np.uint32)
        
        for p in trajectory:
            idx = self._to_cell(p[0], p[1])
            iy = idx // self.grid_size_x
            ix = idx % self.grid_size_x
            grid[iy, ix] += 1
        
        return grid


@dataclass
class PipelineResult:
    """파이프라인 결과 데이터"""
    # 이미지들
    original: np.ndarray
    display: np.ndarray
    skeleton: np.ndarray
    corrected: np.ndarray
    normalized: np.ndarray      # 정규화된 스켈레톤 (모서리 정렬 + 대칭화)
    theoretical: np.ndarray
    
    # 레이아웃 정보
    display_bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    scale: Tuple[float, float]
    offset: Tuple[int, int]
    
    # 메트릭
    metrics: Dict[str, Any]


class LissajousPipeline:
    """리사쥬 이미지 분석 통합 파이프라인"""
    
    def __init__(self, image_path: str, target_size: int = 500):
        """
        Args:
            image_path: 이미지 파일 경로 (HEIF/HEIC/PNG/JPG 지원)
            target_size: 분석용 리사이즈 크기
        """
        self.image_path = Path(image_path)
        self.target_size = target_size
        
        # 중간 결과 저장
        self._original = None
        self._display = None
        self._display_bbox = None
        self._resized = None
        self._binary = None
        self._skeleton = None
        self._corrected = None
        self._center = None
        self._scale = None
        self._offset = None
    
    # =========================================================================
    # Step 1: HEIF 이미지 로드
    # =========================================================================
    def load_image(self) -> np.ndarray:
        """
        Step 1: HEIF/HEIC 포함 다양한 포맷의 이미지 로드
        
        Returns:
            np.ndarray: BGR 이미지
        """
        ext = self.image_path.suffix.lower()
        
        if ext in ['.heif', '.heic', '.hif']:
            with Image.open(self.image_path) as img:
                self._original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            self._original = cv2.imread(str(self.image_path))
        
        if self._original is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {self.image_path}")
        
        return self._original
    
    # =========================================================================
    # Step 2: 디스플레이 영역 검출
    # =========================================================================
    def detect_display(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Step 2: 오실로스코프 디스플레이 영역 자동 검출
        
        Returns:
            Tuple[np.ndarray, tuple]: (크롭된 디스플레이 이미지, 바운딩박스)
        """
        if self._original is None:
            self.load_image()
        
        image = self._original
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 다양한 임계값 시도
        for thresh_val in [50, 80, 100]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                if area < h * w * 0.01 or area > h * w * 0.9:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 99
                if aspect < 3:
                    self._display = image[y:y+bh, x:x+bw]
                    self._display_bbox = (x, y, bw, bh)
                    return self._display, self._display_bbox
        
        # 폴백: 여백 제거
        margin = int(min(h, w) * 0.1)
        self._display = image[margin:h-margin, margin:w-margin]
        self._display_bbox = (margin, margin, w-2*margin, h-2*margin)
        
        return self._display, self._display_bbox
    
    # =========================================================================
    # Step 3: 곡선 추출 & 골격화
    # =========================================================================
    def extract_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: 리사쥬 곡선 추출 및 골격화
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (이진 이미지, 골격화 이미지)
        """
        if self._display is None:
            self.detect_display()
        
        # 분석용 리사이즈
        scale = self.target_size / max(self._display.shape[:2])
        self._resized = cv2.resize(
            self._display, 
            (int(self._display.shape[1] * scale), int(self._display.shape[0] * scale))
        )
        
        # 적응형 이진화
        gray = cv2.cvtColor(self._resized, cv2.COLOR_BGR2GRAY)
        self._binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5
        )
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        self._binary = cv2.morphologyEx(self._binary, cv2.MORPH_OPEN, kernel)
        self._binary = cv2.morphologyEx(self._binary, cv2.MORPH_CLOSE, kernel)
        
        # 골격화
        self._skeleton = (morphology.skeletonize(self._binary > 0) * 255).astype(np.uint8)
        
        return self._binary, self._skeleton
    
    # =========================================================================
    # Step 4: 레이아웃 보정
    # =========================================================================
    def correct_layout(self) -> Dict[str, Any]:
        """
        Step 4: 중심 교점을 프레임 중앙으로 보정
        
        Returns:
            dict: center, scale, offset 정보
        """
        if self._skeleton is None:
            self.extract_curve()
        
        h, w = self._skeleton.shape
        frame_center = (w // 2, h // 2)
        
        # 교차점 검출 (8-연결 이웃 수가 3 이상인 점)
        neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D((self._skeleton > 0).astype(np.uint8), -1, neighbor_kernel)
        intersection_mask = np.logical_and(self._skeleton > 0, neighbor_count >= 3)
        points = np.where(intersection_mask)
        
        # 중앙에 가장 가까운 교차점 찾기
        min_dist = float('inf')
        center_int = frame_center
        for y_pt, x_pt in zip(points[0], points[1]):
            dist = np.sqrt((x_pt - frame_center[0])**2 + (y_pt - frame_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                center_int = (x_pt, y_pt)
        
        # 보정 (교차점 → 프레임 중앙)
        dx = frame_center[0] - center_int[0]
        dy = frame_center[1] - center_int[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        self._corrected = cv2.warpAffine(self._skeleton, M, (w, h))
        
        # 스케일 계산 (보정된 곡선 기준)
        pts = np.where(self._corrected > 0)
        if len(pts[0]) > 0:
            x_min, x_max = pts[1].min(), pts[1].max()
            y_min, y_max = pts[0].min(), pts[0].max()
            scale_x = (x_max - x_min) / 2
            scale_y = (y_max - y_min) / 2
        else:
            scale_x = scale_y = 100
        
        self._center = frame_center
        self._scale = (scale_x, scale_y)
        self._offset = (dx, dy)
        
        return {
            'center': self._center,
            'scale': self._scale,
            'offset': self._offset
        }
    
    # =========================================================================
    # Step 4.5: 스켈레톤 정규화 (완전 대칭 + 모서리 정렬)
    # =========================================================================
    def normalize_skeleton(self, line_thickness: int = 2) -> np.ndarray:
        """
        Step 4.5: 스켈레톤을 이론 곡선과 정확히 비교할 수 있도록 정규화
        
        1. 모든 점을 중심점 기준으로 완전 대칭화 (점 대칭)
        2. 4개 모서리를 이론 곡선 범위와 일치
        3. 선 굵기 통일
        
        Args:
            line_thickness: 출력 선 굵기 (기본 2)
        
        Returns:
            np.ndarray: 정규화된 스켈레톤 이미지 (완전 대칭)
        """
        if self._corrected is None:
            self.correct_layout()
        
        h, w = self._corrected.shape
        cx, cy = self._center
        
        # 1. 스켈레톤 점들 추출
        pts = np.where(self._corrected > 0)
        if len(pts[0]) == 0:
            self._normalized = self._corrected.copy()
            return self._normalized
        
        y_coords, x_coords = pts[0], pts[1]
        
        # 2. 현재 스켈레톤의 경계 (대칭화 전)
        ext_x_min, ext_x_max = x_coords.min(), x_coords.max()
        ext_y_min, ext_y_max = y_coords.min(), y_coords.max()
        
        # 현재 중심에서의 최대 반폭 (for scaling)
        ext_half_w = max(cx - ext_x_min, ext_x_max - cx)
        ext_half_h = max(cy - ext_y_min, ext_y_max - cy)
        
        # 목표 반폭 (scale은 corrected skeleton기준으로 이미 계산됨)
        target_half_w = self._scale[0]
        target_half_h = self._scale[1]
        
        # 스케일 비율
        scale_ratio_x = target_half_w / ext_half_w if ext_half_w > 0 else 1.0
        scale_ratio_y = target_half_h / ext_half_h if ext_half_h > 0 else 1.0
        
        # 3. 좌표 변환 (스케일링)
        scaled_x = (x_coords - cx) * scale_ratio_x + cx
        scaled_y = (y_coords - cy) * scale_ratio_y + cy
        
        # 4. 완전 대칭화: 각 점에 대해 중심 대칭점도 추가
        # (x, y) → 점대칭 → (2*cx - x, 2*cy - y)
        sym_x = 2 * cx - scaled_x
        sym_y = 2 * cy - scaled_y
        
        # 원점과 대칭점 모두 결합
        all_x = np.concatenate([scaled_x, sym_x]).astype(np.int32)
        all_y = np.concatenate([scaled_y, sym_y]).astype(np.int32)
        
        # 5. 새 이미지에 그리기
        self._normalized = np.zeros((h, w), dtype=np.uint8)
        
        # 유효 범위 필터링
        valid = (all_x >= 0) & (all_x < w) & (all_y >= 0) & (all_y < h)
        valid_x = all_x[valid]
        valid_y = all_y[valid]
        
        # 점들을 그리기 (선 굵기 적용)
        if len(valid_x) > 0:
            for px, py in zip(valid_x, valid_y):
                cv2.circle(self._normalized, (int(px), int(py)), 
                          max(1, line_thickness // 2), 255, -1)
        
        # 6. 정보 저장
        self._normalized_info = {
            'scale_ratio': (scale_ratio_x, scale_ratio_y),
            'original_points': len(x_coords),
            'symmetric_points': len(all_x),
            'ext_range': ((ext_x_min, ext_x_max), (ext_y_min, ext_y_max)),
            'target_range': ((cx - target_half_w, cx + target_half_w), 
                            (cy - target_half_h, cy + target_half_h))
        }
        
        return self._normalized
    
    def get_normalized_info(self) -> Dict[str, Any]:
        """정규화 정보 반환"""
        if hasattr(self, '_normalized_info'):
            return self._normalized_info
        return {}
    
    # =========================================================================
    # Step 5: 지표 계산
    # =========================================================================
    def calculate_metrics(self, fx: int, fy: int, phase_deg: float, 
                          duration: float = 0.01, grid_size: int = 256,
                          fov_x: float = 10.0, fov_y: float = 10.0) -> Dict[str, Any]:
        """
        Step 5: 이론 곡선 생성 및 지표 계산
        
        4가지 핵심 지표 (C++ analyzer와 동일):
        - FF (Fill Factor): 채워진 픽셀 비율
        - CV (Coefficient of Variation): 균일도
        - H (Shannon Entropy): 분포 엔트로피
        - max_h: 최대 공백 크기
        
        Args:
            fx, fy: 주파수 (Hz)
            phase_deg: 위상차 (도)
            duration: 시뮬레이션 시간 (초)
            grid_size: 그리드 해상도 (기본 256x256)
            fov_x, fov_y: FOV 범위 (도)
        
        Returns:
            dict: 모든 메트릭 포함
        """
        if self._corrected is None:
            self.correct_layout()
        
        # 정규화된 스켈레톤이 없으면 생성
        if not hasattr(self, '_normalized') or self._normalized is None:
            self.normalize_skeleton()
        
        # ==== 그리드 기반 4가지 핵심 지표 계산 ====
        calc = MetricsCalculator(grid_size=grid_size, fov_x=fov_x, fov_y=fov_y)
        grid_metrics = calc.calculate_full_stats(fx, fy, phase_deg, duration)
        trajectory = calc.generate_trajectory(fx, fy, phase_deg, duration)
        grid_image = calc.get_grid_image(trajectory)
        
        # ==== 이미지 기반 비교 메트릭 (정규화된 스켈레톤 사용) ====
        theoretical = self._generate_theoretical(fx, fy, phase_deg, duration)
        
        kernel = np.ones((5, 5), np.uint8)
        norm_dilated = cv2.dilate(self._normalized, kernel)
        theo_dilated = cv2.dilate(theoretical, kernel)
        
        norm_pixels = (self._normalized > 0).sum()
        theo_pixels = (theoretical > 0).sum()
        
        norm_in_theo = np.logical_and(self._normalized > 0, theo_dilated > 0).sum()
        theo_in_norm = np.logical_and(theoretical > 0, norm_dilated > 0).sum()
        
        fill_rate_extracted = norm_in_theo / norm_pixels if norm_pixels > 0 else 0
        fill_rate_theoretical = theo_in_norm / theo_pixels if theo_pixels > 0 else 0
        
        intersection = np.logical_and(norm_dilated > 0, theo_dilated > 0).sum()
        union = np.logical_or(norm_dilated > 0, theo_dilated > 0).sum()
        iou = intersection / union if union > 0 else 0
        
        return {
            # 4가지 핵심 지표 (그리드 기반)
            'FF': grid_metrics.fill_factor,
            'CV': grid_metrics.cv,
            'entropy': grid_metrics.entropy,
            'max_h': grid_metrics.max_h,
            
            # 그리드 정보
            'grid_size': grid_metrics.grid_size,
            'total_pixels': grid_metrics.total_pixels,
            'filled_pixels': grid_metrics.filled_pixels,
            'grid_image': grid_image,
            'trajectory_points': len(trajectory),
            
            # 이미지 비교 메트릭 (정규화된 스켈레톤 기준)
            'normalized_pixels': int(norm_pixels),
            'theoretical_pixels': int(theo_pixels),
            'fill_rate_extracted': fill_rate_extracted,
            'fill_rate_theoretical': fill_rate_theoretical,
            'combined_fill_rate': (fill_rate_extracted + fill_rate_theoretical) / 2,
            'iou': iou,
            'theoretical_image': theoretical
        }
    
    def _generate_theoretical(self, fx: int, fy: int, phase_deg: float, 
                               duration: float) -> np.ndarray:
        """이론 리사쥬 곡선 생성"""
        phase = np.radians(phase_deg)
        cx, cy = self._center
        scale_x, scale_y = self._scale
        h, w = self._corrected.shape
        
        # 샘플링
        num_samples = max(int(max(fx, fy) * duration * 100), 10000)
        t = np.linspace(0, duration, num_samples)
        
        # 리사쥬 방정식
        x = np.sin(2 * np.pi * fx * t)
        y = np.sin(2 * np.pi * fy * t + phase)
        
        # 이미지 좌표 변환
        x_img = (x * scale_x + cx).astype(np.int32)
        y_img = (y * scale_y + cy).astype(np.int32)
        
        # 이미지 생성
        img = np.zeros((h, w), dtype=np.uint8)
        valid = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
        points = np.column_stack((x_img[valid], y_img[valid])).reshape(-1, 1, 2)
        
        if len(points) > 1:
            cv2.polylines(img, [points], False, 255, 2, cv2.LINE_AA)
        
        return img
    
    # =========================================================================
    # 전체 파이프라인 실행
    # =========================================================================
    def run(self, fx: int, fy: int, phase_deg: float, 
            duration: float = 0.01, visualize: bool = True,
            output_dir: str = './results') -> PipelineResult:
        """
        전체 파이프라인 실행
        
        Args:
            fx, fy: 주파수 (Hz)
            phase_deg: 위상차 (도)
            duration: 시뮬레이션 시간 (초)
            visualize: 시각화 여부
            output_dir: 결과 저장 디렉토리
        
        Returns:
            PipelineResult: 전체 결과
        """
        print(f"\n{'='*60}")
        print(f"Lissajous Pipeline: {self.image_path.name}")
        print(f"  x(t) = A·sin(2π·{fx}·t)")
        print(f"  y(t) = B·sin(2π·{fy}·t + {phase_deg}°)")
        print(f"  Duration: {duration*1000:.1f}ms")
        print(f"{'='*60}")
        
        # Step 1
        print("\n[1/5] Loading image...")
        self.load_image()
        print(f"  → Original: {self._original.shape[1]}x{self._original.shape[0]}")
        
        # Step 2
        print("[2/5] Detecting display region...")
        self.detect_display()
        print(f"  → Display: {self._display.shape[1]}x{self._display.shape[0]}")
        
        # Step 3
        print("[3/5] Extracting curve & skeletonization...")
        self.extract_curve()
        skeleton_pixels = (self._skeleton > 0).sum()
        print(f"  → Skeleton: {skeleton_pixels:,} pixels")
        
        # Step 4
        print("[4/6] Correcting layout...")
        layout = self.correct_layout()
        print(f"  → Center: {layout['center']}")
        print(f"  → Scale: ({layout['scale'][0]:.1f}, {layout['scale'][1]:.1f})")
        print(f"  → Offset: {layout['offset']}")
        
        # Step 4.5: 스켈레톤 정규화
        print("[5/6] Normalizing skeleton...")
        self.normalize_skeleton(line_thickness=2)
        norm_info = self.get_normalized_info()
        print(f"  → Scale Ratio: ({norm_info['scale_ratio'][0]:.3f}, {norm_info['scale_ratio'][1]:.3f})")
        print(f"  → Aligned to theoretical corners")
        
        # Step 5
        print("[6/6] Calculating metrics...")
        metrics = self.calculate_metrics(fx, fy, phase_deg, duration)
        
        print(f"\n📊 Core Metrics (grid {metrics['grid_size'][0]}x{metrics['grid_size'][1]}):")
        print(f"  FF (Fill Factor):    {metrics['FF']*100:.2f}% ({metrics['filled_pixels']:,}/{metrics['total_pixels']:,} pixels)")
        print(f"  CV (Uniformity):     {metrics['CV']:.4f}")
        print(f"  H (Entropy):         {metrics['entropy']:.4f}")
        print(f"  max_h (Max Gap):     {metrics['max_h']:.6f}°")
        print(f"  Trajectory Points:   {metrics['trajectory_points']:,}")
        
        print(f"\n📐 Image Comparison (Normalized):")
        print(f"  Extracted → Theory:  {metrics['fill_rate_extracted']*100:.1f}%")
        print(f"  Theory → Extracted:  {metrics['fill_rate_theoretical']*100:.1f}%")
        print(f"  IoU:                 {metrics['iou']*100:.1f}%")
        
        # 결과 객체 생성
        result = PipelineResult(
            original=self._original,
            display=self._display,
            skeleton=self._skeleton,
            corrected=self._corrected,
            normalized=self._normalized,
            theoretical=metrics['theoretical_image'],
            display_bbox=self._display_bbox,
            center=self._center,
            scale=self._scale,
            offset=self._offset,
            metrics=metrics
        )
        
        # 시각화
        if visualize:
            self._visualize(result, fx, fy, phase_deg, duration, output_dir)
        
        return result
    
    def _visualize(self, result: PipelineResult, fx: int, fy: int, 
                   phase_deg: float, duration: float, output_dir: str):
        """결과 시각화 및 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1
        axes[0, 0].imshow(cv2.cvtColor(self._resized, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Cropped Display')
        axes[0, 0].axis('off')
        
        # Center-aligned Skeleton (보정만)
        axes[0, 1].imshow(result.corrected, cmap='gray')
        axes[0, 1].scatter(*result.center, c='red', s=50, marker='+', linewidths=2)
        axes[0, 1].set_title(f'Center Aligned\nOffset: {result.offset}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(result.theoretical, cmap='gray')
        axes[0, 2].scatter(*result.center, c='red', s=50, marker='+', linewidths=2)
        axes[0, 2].set_title(f'Theoretical ({duration*1000:.0f}ms)\n{fx}:{fy} Hz, φ={phase_deg}°')
        axes[0, 2].axis('off')
        
        # Row 2: 정규화된 대칭 스켈레톤 with 그리드 배경
        h, w = result.normalized.shape
        cx, cy = result.center
        
        # 그리드 배경 생성 (밝은 회색)
        grid_bg = np.ones((h, w), dtype=np.uint8) * 30  # 어두운 배경
        
        # 그리드 라인 그리기
        grid_spacing = 20
        for i in range(0, w, grid_spacing):
            cv2.line(grid_bg, (i, 0), (i, h), 50, 1)  # 세로선
        for j in range(0, h, grid_spacing):
            cv2.line(grid_bg, (0, j), (w, j), 50, 1)  # 가로선
        
        # 중심선 (밝게)
        cv2.line(grid_bg, (cx, 0), (cx, h), 80, 1)
        cv2.line(grid_bg, (0, cy), (w, cy), 80, 1)
        
        # 스켈레톤 오버레이
        grid_with_skeleton = cv2.cvtColor(grid_bg, cv2.COLOR_GRAY2RGB)
        grid_with_skeleton[:, :, 1] = np.where(result.normalized > 0, 255, grid_with_skeleton[:, :, 1])
        
        axes[1, 0].imshow(grid_with_skeleton)
        axes[1, 0].scatter(cx, cy, c='red', s=50, marker='+', linewidths=2)
        norm_info = self.get_normalized_info()
        axes[1, 0].set_title(f'Symmetric Normalized (Point Symmetric)\n{norm_info.get("symmetric_points", 0):,} pts')
        axes[1, 0].axis('off')
        
        # 매칭 분석 (정규화된 스켈레톤 사용)
        kernel = np.ones((5, 5), np.uint8)
        theo_d = cv2.dilate(result.theoretical, kernel)
        norm_d = cv2.dilate(result.normalized, kernel)
        
        matched = np.zeros((*result.normalized.shape, 3), dtype=np.uint8)
        both = np.logical_and(result.normalized > 0, theo_d > 0)
        norm_only = np.logical_and(result.normalized > 0, theo_d == 0)
        theo_only = np.logical_and(result.theoretical > 0, norm_d == 0)
        
        matched[:, :, 0] = np.where(both, 255, 0)
        matched[:, :, 1] = np.where(both, 255, 0)
        matched[:, :, 1] = np.where(norm_only, 255, matched[:, :, 1])
        matched[:, :, 2] = np.where(theo_only, 255, 0)
        
        axes[1, 1].imshow(matched)
        axes[1, 1].set_title('Match (Y=both, G=norm only, B=theo only)')
        axes[1, 1].axis('off')
        
        # 결과 텍스트
        m = result.metrics
        result_text = f"""Lissajous Pipeline Results

Parameters:
  fx={fx} Hz, fy={fy} Hz
  phase={phase_deg}°, t={duration*1000:.1f}ms

Core Metrics ({m['grid_size'][0]}x{m['grid_size'][1]} grid):
  FF:      {m['FF']*100:.2f}%
  CV:      {m['CV']:.4f}
  H:       {m['entropy']:.4f}
  max_h:   {m['max_h']:.6f}°

Grid Info:
  Filled:  {m['filled_pixels']:,} / {m['total_pixels']:,}
  Points:  {m['trajectory_points']:,}

Image Comparison:
  Ext→Theo: {m['fill_rate_extracted']*100:.1f}%
  Theo→Ext: {m['fill_rate_theoretical']*100:.1f}%
  IoU:      {m['iou']*100:.1f}%"""
        
        axes[1, 2].text(0.05, 0.95, result_text, fontsize=10, verticalalignment='top',
                        family='monospace', transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Results Summary')
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'{self.image_path.stem}_pipeline.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {fig_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Lissajous Pipeline - 통합 이미지 처리 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python lissajous_pipeline.py image.HIF -fx 1350 -fy 1480 -phase 0
  python lissajous_pipeline.py image.png -fx 1000 -fy 1100 -duration 0.02
        """
    )
    
    parser.add_argument('image', help='이미지 파일 경로 (HEIF/HEIC/PNG/JPG)')
    parser.add_argument('-fx', type=int, default=1350, help='X축 주파수 (Hz)')
    parser.add_argument('-fy', type=int, default=1480, help='Y축 주파수 (Hz)')
    parser.add_argument('-phase', type=float, default=0, help='위상차 (도)')
    parser.add_argument('-duration', type=float, default=0.01, help='시뮬레이션 시간 (초)')
    parser.add_argument('-o', '--output', default='./results', help='출력 디렉토리')
    parser.add_argument('--no-viz', action='store_true', help='시각화 비활성화')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ 파일을 찾을 수 없습니다: {args.image}")
        return
    
    pipeline = LissajousPipeline(args.image)
    result = pipeline.run(
        fx=args.fx,
        fy=args.fy,
        phase_deg=args.phase,
        duration=args.duration,
        visualize=not args.no_viz,
        output_dir=args.output
    )
    
    print(f"\n{'='*60}")
    print("✅ Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
리사쥬 곡선 보정 및 비교
1. Extracted curve의 center intersection → frame center로 이동 (보정)
2. 부드러운 이론 궤적 생성
3. 보정된 곡선과 이론 궤적 비교
"""

import os
import numpy as np
import cv2
from pathlib import Path
from skimage import morphology
from scipy import ndimage

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    path = Path(image_path)
    ext = path.suffix.lower()
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


def detect_display_region(image):
    """디스플레이 영역 검출"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
            if max(bw, bh) / min(bw, bh) < 3:
                return image[y:y+bh, x:x+bw], (x, y, bw, bh)
    
    margin = int(min(h, w) * 0.1)
    return image[margin:h-margin, margin:w-margin], (margin, margin, w-2*margin, h-2*margin)


def extract_curve(image):
    """곡선 추출 및 골격화"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -5)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skeleton = (morphology.skeletonize(binary > 0) * 255).astype(np.uint8)
    return binary, skeleton


def find_intersection_points(skeleton):
    """교점 찾기"""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    intersection_mask = np.logical_and(skeleton > 0, neighbor_count >= 3)
    points = np.where(intersection_mask)
    return [(x, y) for y, x in zip(points[0], points[1])]


def find_center_intersection(skeleton):
    """가장 중앙에 가까운 교점"""
    h, w = skeleton.shape
    center_x, center_y = w // 2, h // 2
    intersections = find_intersection_points(skeleton)
    
    if not intersections:
        points = np.where(skeleton > 0)
        if len(points[0]) > 0:
            return int(np.mean(points[1])), int(np.mean(points[0]))
        return center_x, center_y
    
    min_dist = float('inf')
    center_int = (center_x, center_y)
    for x, y in intersections:
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_int = (x, y)
    return center_int


def correct_curve_alignment(skeleton, center_intersection, frame_center):
    """
    추출된 곡선을 보정: center_intersection을 frame_center로 이동
    
    Args:
        skeleton: 골격화된 추출 곡선
        center_intersection: 현재 중앙 교점 (x, y)
        frame_center: 목표 프레임 중앙 (x, y)
    
    Returns:
        corrected: 보정된 이미지
        offset: 적용된 오프셋 (dx, dy)
    """
    h, w = skeleton.shape
    
    # 이동량 계산
    dx = frame_center[0] - center_intersection[0]
    dy = frame_center[1] - center_intersection[1]
    
    # 아핀 변환 행렬 (이동만)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # 이동 적용
    corrected = cv2.warpAffine(skeleton, M, (w, h), borderValue=0)
    
    return corrected, (dx, dy)


def get_curve_scale(skeleton):
    """곡선의 반경(스케일) 계산"""
    points = np.where(skeleton > 0)
    if len(points[0]) == 0:
        return 100
    
    center_y, center_x = np.mean(points[0]), np.mean(points[1])
    distances = np.sqrt((points[1] - center_x)**2 + (points[0] - center_y)**2)
    return np.max(distances)


def generate_smooth_lissajous(fx, fy, phase_deg, center, scale, img_shape, 
                               duration=None, line_thickness=2):
    """
    부드러운 이론 리사쥬 궤적 생성 (완전한 패턴)
    
    Args:
        fx, fy: 주파수 (Hz)
        phase_deg: 위상차 (도)
        center: 중심점 (x, y)
        scale: 스케일
        img_shape: 이미지 크기
        duration: 시간 (초) - None이면 자동 계산 (완전 패턴)
        line_thickness: 선 두께
    """
    from math import gcd
    
    phase = np.radians(phase_deg)
    
    # 완전한 패턴을 위한 시간 계산
    # GCD(fx, fy) -> 패턴 주기 = 1/GCD 초
    g = gcd(int(fx), int(fy))
    pattern_period = 1.0 / g  # 완전한 패턴 주기
    
    if duration is None:
        duration = pattern_period
    
    # 충분한 샘플 수 (매끄러운 곡선을 위해)
    num_samples = int(max(fx, fy) * duration * 1000)  # 사이클당 1000 샘플
    num_samples = max(num_samples, 50000)  # 최소 50000개
    
    t = np.linspace(0, duration, num_samples)
    
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    cx, cy = center
    x_img = (x * scale + cx).astype(np.float32)
    y_img = (y * scale + cy).astype(np.float32)
    
    # 이미지 생성
    h, w = img_shape[:2]
    theo_img = np.zeros((h, w), dtype=np.uint8)
    
    # polylines로 부드럽게 그리기
    points = np.column_stack((x_img, y_img)).astype(np.int32)
    valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    points = points[valid].reshape(-1, 1, 2)
    
    if len(points) > 1:
        cv2.polylines(theo_img, [points], False, 255, line_thickness, cv2.LINE_AA)
    
    cycles_x = fx * duration
    cycles_y = fy * duration
    
    return theo_img, num_samples, cycles_x, cycles_y, duration


def calculate_metrics(extracted, theoretical):
    """비교 메트릭 계산"""
    kernel = np.ones((5, 5), np.uint8)
    ext_d = cv2.dilate(extracted, kernel)
    theo_d = cv2.dilate(theoretical, kernel)
    
    ext_pixels = (extracted > 0).sum()
    theo_pixels = (theoretical > 0).sum()
    
    ext_in_theo = np.logical_and(extracted > 0, theo_d > 0).sum()
    theo_in_ext = np.logical_and(theoretical > 0, ext_d > 0).sum()
    
    intersection = np.logical_and(ext_d > 0, theo_d > 0).sum()
    union = np.logical_or(ext_d > 0, theo_d > 0).sum()
    
    return {
        'ext_pixels': int(ext_pixels),
        'theo_pixels': int(theo_pixels),
        'ext_to_theo': ext_in_theo / ext_pixels if ext_pixels > 0 else 0,
        'theo_to_ext': theo_in_ext / theo_pixels if theo_pixels > 0 else 0,
        'iou': intersection / union if union > 0 else 0
    }


def analyze_and_correct(image_path, fx=1350, fy=1480, phase=0, 
                        duration=0.01, output_dir='./results'):
    """보정 및 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*65}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"Parameters: fx={fx}Hz, fy={fy}Hz, phase={phase}°, duration={duration*1000:.0f}ms")
    print(f"{'='*65}")
    
    # 로드
    original = load_image(image_path)
    if original is None:
        return None
    
    # 디스플레이 검출
    display, bbox = detect_display_region(original)
    disp_h, disp_w = display.shape[:2]
    
    # 리사이즈
    target_size = 500
    scale_factor = target_size / max(disp_h, disp_w)
    resized = cv2.resize(display, (int(disp_w * scale_factor), int(disp_h * scale_factor)))
    res_h, res_w = resized.shape[:2]
    
    # 곡선 추출
    binary, skeleton = extract_curve(resized)
    
    # 중앙 교점 & 프레임 중앙
    center_intersection = find_center_intersection(skeleton)
    frame_center = (res_w // 2, res_h // 2)
    
    print(f"Center intersection: {center_intersection}")
    print(f"Frame center: {frame_center}")
    
    # ========== 보정 ==========
    corrected, offset = correct_curve_alignment(skeleton, center_intersection, frame_center)
    print(f"Correction offset: dx={offset[0]}, dy={offset[1]}")
    
    # 보정 후 스케일 계산
    curve_scale = get_curve_scale(corrected)
    print(f"Curve scale: {curve_scale:.1f}px")
    
    # ========== 이론 궤적 (0.01초, 매끄럽게) ==========
    theo_img, num_samples, cycles_x, cycles_y, actual_duration = generate_smooth_lissajous(
        fx, fy, phase, frame_center, curve_scale, corrected.shape, 
        duration=0.01, line_thickness=2  # 0.01초 누적시간
    )
    print(f"Theoretical: {num_samples:,} samples, {cycles_x:.1f}x{cycles_y:.1f} cycles, {actual_duration*1000:.1f}ms")
    
    # ========== 메트릭 ==========
    # 보정 전
    metrics_before = calculate_metrics(skeleton, theo_img)
    # 보정 후
    metrics_after = calculate_metrics(corrected, theo_img)
    
    print(f"\nBefore correction:")
    print(f"  Ext→Theo: {metrics_before['ext_to_theo']*100:.1f}%, Theo→Ext: {metrics_before['theo_to_ext']*100:.1f}%")
    print(f"After correction:")
    print(f"  Ext→Theo: {metrics_after['ext_to_theo']*100:.1f}%, Theo→Ext: {metrics_after['theo_to_ext']*100:.1f}%")
    
    # ========== 시각화 ==========
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: 추출 과정
    axes[0, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cropped Display')
    axes[0, 0].axis('off')
    
    # 원본 skeleton + 교점
    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].scatter(*center_intersection, c='yellow', s=150, marker='o', 
                       edgecolors='red', linewidths=2, label='Center Int.')
    axes[0, 1].scatter(*frame_center, c='cyan', s=150, marker='+', 
                       linewidths=3, label='Frame Center')
    # 오프셋 화살표
    axes[0, 1].annotate('', xy=frame_center, xytext=center_intersection,
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].set_title(f'Original Extracted\nOffset needed: ({offset[0]}, {offset[1]})')
    axes[0, 1].axis('off')
    
    # 보정된 skeleton
    axes[0, 2].imshow(corrected, cmap='gray')
    corrected_center_int = (center_intersection[0] + offset[0], center_intersection[1] + offset[1])
    axes[0, 2].scatter(*corrected_center_int, c='yellow', s=150, marker='o', 
                       edgecolors='red', linewidths=2)
    axes[0, 2].scatter(*frame_center, c='cyan', s=150, marker='+', linewidths=3)
    axes[0, 2].set_title('Corrected Extracted\n(Aligned to center)')
    axes[0, 2].axis('off')
    
    # 이론 궤적
    axes[0, 3].imshow(theo_img, cmap='gray')
    axes[0, 3].scatter(*frame_center, c='red', s=50, marker='+')
    axes[0, 3].set_title(f'Theoretical ({duration*1000:.0f}ms)\n{num_samples} samples, smooth curve')
    axes[0, 3].axis('off')
    
    # Row 2: 비교
    # 보정 전 오버레이
    overlay_before = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay_before[:, :, 0] = theo_img
    overlay_before[:, :, 1] = skeleton
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title(f'Before Correction\nExt→Theo: {metrics_before["ext_to_theo"]*100:.1f}%')
    axes[1, 0].axis('off')
    
    # 보정 후 오버레이
    overlay_after = np.zeros((*corrected.shape, 3), dtype=np.uint8)
    overlay_after[:, :, 0] = theo_img
    overlay_after[:, :, 1] = corrected
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title(f'After Correction\nExt→Theo: {metrics_after["ext_to_theo"]*100:.1f}%')
    axes[1, 1].axis('off')
    
    # 매칭 분석 (보정 후)
    kernel = np.ones((5, 5), np.uint8)
    theo_d = cv2.dilate(theo_img, kernel)
    ext_d = cv2.dilate(corrected, kernel)
    
    matched = np.zeros((*corrected.shape, 3), dtype=np.uint8)
    both = np.logical_and(corrected > 0, theo_d > 0)
    ext_only = np.logical_and(corrected > 0, theo_d == 0)
    theo_only = np.logical_and(theo_img > 0, ext_d == 0)
    
    matched[:, :, 0] = np.where(both, 255, 0)  # Yellow = matched
    matched[:, :, 1] = np.where(both, 255, 0)
    matched[:, :, 1] = np.where(ext_only, 255, matched[:, :, 1])  # Green = ext only
    matched[:, :, 2] = np.where(theo_only, 255, 0)  # Blue = theo only
    
    axes[1, 2].imshow(matched)
    axes[1, 2].set_title('Match Analysis (corrected)\nY=match, G=ext, B=theo')
    axes[1, 2].axis('off')
    
    # 결과
    result_text = f"""Correction Analysis

Original:
  Center intersection: {center_intersection}
  Frame center: {frame_center}
  Offset applied: ({offset[0]}, {offset[1]})

Theoretical ({duration*1000:.0f}ms):
  Samples: {num_samples:,}
  X cycles: {cycles_x:.2f}
  Y cycles: {cycles_y:.2f}

Before Correction:
  Ext→Theo: {metrics_before['ext_to_theo']*100:.1f}%
  Theo→Ext: {metrics_before['theo_to_ext']*100:.1f}%
  IoU: {metrics_before['iou']*100:.1f}%

After Correction:
  Ext→Theo: {metrics_after['ext_to_theo']*100:.1f}%
  Theo→Ext: {metrics_after['theo_to_ext']*100:.1f}%
  IoU: {metrics_after['iou']*100:.1f}%

Improvement:
  Ext→Theo: +{(metrics_after['ext_to_theo']-metrics_before['ext_to_theo'])*100:.1f}%
  Theo→Ext: +{(metrics_after['theo_to_ext']-metrics_before['theo_to_ext'])*100:.1f}%"""
    
    axes[1, 3].text(0.05, 0.95, result_text, fontsize=9, verticalalignment='top',
                    family='monospace', transform=axes[1, 3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 3].axis('off')
    axes[1, 3].set_title('Results')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    fig_path = os.path.join(output_dir, f'{base_name}_corrected.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close()
    
    return {
        'offset': offset,
        'before': metrics_before,
        'after': metrics_after
    }


if __name__ == '__main__':
    fx, fy, phase = 1350, 1480, 0
    duration = 0.01
    
    results = []
    for img in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img):
            r = analyze_and_correct(img, fx, fy, phase, duration)
            if r:
                results.append((Path(img).name, r))
    
    print("\n" + "="*65)
    print("Summary")
    print("="*65)
    for name, r in results:
        before = r['before']
        after = r['after']
        print(f"\n{name}:")
        print(f"  Offset: {r['offset']}")
        print(f"  Before: Ext→Theo {before['ext_to_theo']*100:.1f}%, IoU {before['iou']*100:.1f}%")
        print(f"  After:  Ext→Theo {after['ext_to_theo']*100:.1f}%, IoU {after['iou']*100:.1f}%")
        print(f"  Improvement: +{(after['iou']-before['iou'])*100:.1f}% IoU")

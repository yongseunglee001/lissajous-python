#!/usr/bin/env python3
"""
Corrected Extracted 그리드 위에 정확한 리사쥬 궤적 시뮬레이션
x(t) = A * sin(2π * fx * t)
y(t) = B * sin(2π * fy * t + φ)
"""

import os
import numpy as np
import cv2
from pathlib import Path
from skimage import morphology

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


def detect_display(image):
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


def extract_and_correct(image):
    """추출 및 보정"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -5)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skeleton = (morphology.skeletonize(binary > 0) * 255).astype(np.uint8)
    
    # 교점 찾기
    h, w = skeleton.shape
    center_x, center_y = w // 2, h // 2
    
    neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, neighbor_kernel)
    intersection_mask = np.logical_and(skeleton > 0, neighbor_count >= 3)
    points = np.where(intersection_mask)
    
    # 중앙 교점 찾기
    min_dist = float('inf')
    center_int = (center_x, center_y)
    for y_pt, x_pt in zip(points[0], points[1]):
        dist = np.sqrt((x_pt - center_x)**2 + (y_pt - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_int = (x_pt, y_pt)
    
    # 보정 (center_int → frame_center)
    dx = center_x - center_int[0]
    dy = center_y - center_int[1]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    corrected = cv2.warpAffine(skeleton, M, (w, h))
    
    # 스케일 계산 (보정된 곡선 기준)
    pts = np.where(corrected > 0)
    if len(pts[0]) > 0:
        # X, Y 각각의 범위
        x_min, x_max = pts[1].min(), pts[1].max()
        y_min, y_max = pts[0].min(), pts[0].max()
        scale_x = (x_max - x_min) / 2
        scale_y = (y_max - y_min) / 2
    else:
        scale_x = scale_y = 100
    
    return skeleton, corrected, (center_x, center_y), (scale_x, scale_y), (dx, dy)


def simulate_lissajous(fx, fy, phase_deg, center, scale, img_shape, duration=0.01):
    """
    리사쥬 방정식으로 궤적 시뮬레이션
    
    x(t) = scale_x * sin(2π * fx * t)
    y(t) = scale_y * sin(2π * fy * t + φ)
    
    Args:
        fx, fy: 주파수 (Hz)
        phase_deg: 위상차 (도)
        center: 중심점 (cx, cy)
        scale: 스케일 (scale_x, scale_y)
        img_shape: 이미지 크기 (h, w)
        duration: 시뮬레이션 시간 (초)
    
    Returns:
        이미지, 샘플 수
    """
    phase = np.radians(phase_deg)
    cx, cy = center
    scale_x, scale_y = scale
    h, w = img_shape[:2]
    
    # 시간 배열 (0.01초 동안)
    # 충분한 샘플링: 가장 높은 주파수의 100배
    num_samples = int(max(fx, fy) * duration * 100)
    num_samples = max(num_samples, 10000)
    
    t = np.linspace(0, duration, num_samples)
    
    # 리사쥬 방정식
    # x(t) = sin(2π * fx * t)
    # y(t) = sin(2π * fy * t + φ)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표로 변환 (중심 + 스케일)
    x_img = (x * scale_x + cx).astype(np.int32)
    y_img = (y * scale_y + cy).astype(np.int32)
    
    # 이미지에 그리기
    img = np.zeros((h, w), dtype=np.uint8)
    
    # 유효한 점들만 필터링
    valid = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    x_valid = x_img[valid]
    y_valid = y_img[valid]
    
    # 선 그리기 (연결된 점들)
    points = np.column_stack((x_valid, y_valid)).reshape(-1, 1, 2)
    if len(points) > 1:
        cv2.polylines(img, [points], False, 255, 2, cv2.LINE_AA)
    
    return img, num_samples


def main(image_path, fx=1350, fy=1480, phase=0, duration=0.01, output_dir='./results'):
    """메인 함수"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Simulating Lissajous on: {Path(image_path).name}")
    print(f"  x(t) = A·sin(2π·{fx}·t)")
    print(f"  y(t) = B·sin(2π·{fy}·t + {phase}°)")
    print(f"  Duration: {duration*1000:.1f}ms")
    print(f"{'='*60}")
    
    # 로드
    original = load_image(image_path)
    if original is None:
        print("Load failed")
        return None
    
    # 디스플레이 검출
    display, bbox = detect_display(original)
    
    # 리사이즈
    target_size = 500
    scale_factor = target_size / max(display.shape[:2])
    resized = cv2.resize(display, (int(display.shape[1] * scale_factor), 
                                    int(display.shape[0] * scale_factor)))
    
    # 추출 및 보정
    skeleton, corrected, center, scale, offset = extract_and_correct(resized)
    
    print(f"Frame center: {center}")
    print(f"Scale (X, Y): ({scale[0]:.1f}, {scale[1]:.1f})")
    print(f"Correction offset: ({offset[0]}, {offset[1]})")
    
    # 리사쥬 시뮬레이션
    theo_img, num_samples = simulate_lissajous(
        fx, fy, phase, center, scale, corrected.shape, duration
    )
    print(f"Simulated: {num_samples:,} samples")
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cropped Display')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corrected, cmap='gray')
    axes[0, 1].scatter(*center, c='red', s=50, marker='+', linewidths=2)
    axes[0, 1].set_title(f'Corrected Extracted\nCenter: {center}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(theo_img, cmap='gray')
    axes[0, 2].scatter(*center, c='red', s=50, marker='+', linewidths=2)
    axes[0, 2].set_title(f'Theoretical ({duration*1000:.0f}ms)\nx=sin(2π·{fx}·t), y=sin(2π·{fy}·t)')
    axes[0, 2].axis('off')
    
    # Row 2
    # 오버레이
    overlay = np.zeros((*corrected.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = theo_img      # Red = theory
    overlay[:, :, 1] = corrected     # Green = extracted
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 0].axis('off')
    
    # 매칭 분석
    kernel = np.ones((5, 5), np.uint8)
    theo_d = cv2.dilate(theo_img, kernel)
    ext_d = cv2.dilate(corrected, kernel)
    
    matched = np.zeros((*corrected.shape, 3), dtype=np.uint8)
    both = np.logical_and(corrected > 0, theo_d > 0)
    ext_only = np.logical_and(corrected > 0, theo_d == 0)
    theo_only = np.logical_and(theo_img > 0, ext_d == 0)
    
    matched[:, :, 0] = np.where(both, 255, 0)
    matched[:, :, 1] = np.where(both, 255, 0)
    matched[:, :, 1] = np.where(ext_only, 255, matched[:, :, 1])
    matched[:, :, 2] = np.where(theo_only, 255, 0)
    
    axes[1, 1].imshow(matched)
    axes[1, 1].set_title('Match (Y=both, G=ext, B=theo)')
    axes[1, 1].axis('off')
    
    # 메트릭
    ext_pixels = (corrected > 0).sum()
    theo_pixels = (theo_img > 0).sum()
    ext_in_theo = np.logical_and(corrected > 0, theo_d > 0).sum()
    theo_in_ext = np.logical_and(theo_img > 0, ext_d > 0).sum()
    
    ext_to_theo = ext_in_theo / ext_pixels if ext_pixels > 0 else 0
    theo_to_ext = theo_in_ext / theo_pixels if theo_pixels > 0 else 0
    
    result_text = f"""Lissajous Simulation

Equation:
  x(t) = {scale[0]:.0f}·sin(2π·{fx}·t)
  y(t) = {scale[1]:.0f}·sin(2π·{fy}·t + {phase}°)
  
Center: {center}
Duration: {duration*1000:.1f} ms
Samples: {num_samples:,}

Metrics:
  Ext→Theo: {ext_to_theo*100:.1f}%
  Theo→Ext: {theo_to_ext*100:.1f}%

Pixel counts:
  Extracted: {ext_pixels:,}
  Theoretical: {theo_pixels:,}"""
    
    axes[1, 2].text(0.05, 0.95, result_text, fontsize=10, verticalalignment='top',
                    family='monospace', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    fig_path = os.path.join(output_dir, f'{base_name}_simulated.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close()
    
    return {'ext_to_theo': ext_to_theo, 'theo_to_ext': theo_to_ext}


if __name__ == '__main__':
    fx, fy, phase = 1350, 1480, 0
    duration = 0.01
    
    for img in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img):
            main(img, fx, fy, phase, duration)

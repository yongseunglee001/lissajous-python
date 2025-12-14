#!/usr/bin/env python3
"""
추출된 곡선의 레이아웃 기반 채움률 계산
1. Extracted Curve의 바운딩 박스 계산
2. 그 틀에 맞춰 이론 곡선 생성
3. 채움률(Fill Rate) 계산
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
            aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 99
            if aspect < 3:
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
    
    # 골격화
    skeleton = (morphology.skeletonize(binary > 0) * 255).astype(np.uint8)
    
    return binary, skeleton


def get_curve_layout(skeleton):
    """
    추출된 곡선의 레이아웃(바운딩 박스) 계산
    Returns: (x_min, y_min, x_max, y_max, center_x, center_y, width, height)
    """
    points = np.where(skeleton > 0)
    
    if len(points[0]) == 0:
        h, w = skeleton.shape
        return 0, 0, w, h, w//2, h//2, w, h
    
    y_coords = points[0]
    x_coords = points[1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_min, y_min, x_max, y_max, center_x, center_y, width, height


def generate_lissajous_in_layout(fx, fy, phase_deg, layout, img_shape, num_points=20000):
    """
    추출된 곡선의 레이아웃에 맞춰 이론 곡선 생성
    """
    x_min, y_min, x_max, y_max, center_x, center_y, width, height = layout
    
    phase = np.radians(phase_deg)
    t = np.linspace(0, 1.0, num_points)
    
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 레이아웃에 맞춰 스케일링
    scale_x = width / 2
    scale_y = height / 2
    
    x_img = (x * scale_x + center_x).astype(np.int32)
    y_img = (y * scale_y + center_y).astype(np.int32)
    
    # 이미지 생성
    h, w = img_shape[:2]
    theo_img = np.zeros((h, w), dtype=np.uint8)
    
    points = np.column_stack((x_img, y_img))
    # 범위 체크
    valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    points = points[valid]
    
    for i in range(len(points) - 1):
        cv2.line(theo_img, tuple(points[i]), tuple(points[i+1]), 255, 2)
    
    return theo_img


def calculate_fill_rate(extracted, theoretical):
    """
    채움률 계산
    - 추출된 곡선 중 이론에도 있는 비율
    - 이론 곡선 중 추출에도 있는 비율
    """
    # 팽창 (약간의 오차 허용)
    kernel = np.ones((5, 5), np.uint8)
    ext_dilated = cv2.dilate(extracted, kernel)
    theo_dilated = cv2.dilate(theoretical, kernel)
    
    ext_pixels = (extracted > 0).sum()
    theo_pixels = (theoretical > 0).sum()
    
    # 추출된 곡선 중 이론 곡선에 포함된 비율
    ext_in_theo = np.logical_and(extracted > 0, theo_dilated > 0).sum()
    fill_rate_extracted = ext_in_theo / ext_pixels if ext_pixels > 0 else 0
    
    # 이론 곡선 중 추출된 곡선에 포함된 비율
    theo_in_ext = np.logical_and(theoretical > 0, ext_dilated > 0).sum()
    fill_rate_theoretical = theo_in_ext / theo_pixels if theo_pixels > 0 else 0
    
    # 전체 IoU
    intersection = np.logical_and(ext_dilated > 0, theo_dilated > 0).sum()
    union = np.logical_or(ext_dilated > 0, theo_dilated > 0).sum()
    iou = intersection / union if union > 0 else 0
    
    return {
        'extracted_pixels': int(ext_pixels),
        'theoretical_pixels': int(theo_pixels),
        'fill_rate_extracted': fill_rate_extracted,  # 추출된 곡선의 이론 매칭률
        'fill_rate_theoretical': fill_rate_theoretical,  # 이론 곡선의 추출 매칭률
        'iou': iou,
        'combined_fill_rate': (fill_rate_extracted + fill_rate_theoretical) / 2
    }


def analyze_with_layout(image_path, fx, fy, phase, output_dir='./results'):
    """레이아웃 기반 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"Parameters: fx={fx}Hz, fy={fy}Hz, phase={phase}°")
    print(f"{'='*60}")
    
    # 로드
    original = load_image(image_path)
    if original is None:
        print("Failed")
        return None
    
    # 디스플레이 검출
    display, bbox = detect_display_region(original)
    print(f"Display: {display.shape[1]}x{display.shape[0]}")
    
    # 분석용 리사이즈
    target_size = 500
    scale = target_size / max(display.shape[:2])
    resized = cv2.resize(display, (int(display.shape[1] * scale), int(display.shape[0] * scale)))
    
    # 곡선 추출
    binary, skeleton = extract_curve(resized)
    
    # 레이아웃 계산
    layout = get_curve_layout(skeleton)
    x_min, y_min, x_max, y_max, cx, cy, w, h = layout
    print(f"\nExtracted Curve Layout:")
    print(f"  Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"  Center: ({cx}, {cy})")
    print(f"  Size: {w}x{h}")
    
    # 레이아웃에 맞춘 이론 곡선 생성
    theo_matched = generate_lissajous_in_layout(fx, fy, phase, layout, skeleton.shape)
    
    # 채움률 계산
    fill = calculate_fill_rate(skeleton, theo_matched)
    
    print(f"\nFill Rate Analysis:")
    print(f"  Extracted pixels: {fill['extracted_pixels']:,}")
    print(f"  Theoretical pixels: {fill['theoretical_pixels']:,}")
    print(f"  Extracted -> Theory match: {fill['fill_rate_extracted']*100:.2f}%")
    print(f"  Theory -> Extracted match: {fill['fill_rate_theoretical']*100:.2f}%")
    print(f"  Combined Fill Rate: {fill['combined_fill_rate']*100:.2f}%")
    print(f"  IoU: {fill['iou']*100:.2f}%")
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cropped Display (resized)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(skeleton, cmap='gray')
    # 바운딩 박스 표시
    rect = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=2)
    axes[0, 1].add_patch(rect)
    axes[0, 1].plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
    axes[0, 1].set_title(f'Extracted Curve\nLayout: {w}x{h}, Center: ({cx},{cy})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(theo_matched, cmap='gray')
    axes[0, 2].set_title(f'Theory (matched to layout)\n{fx}:{fy}Hz, phase={phase}°')
    axes[0, 2].axis('off')
    
    # Row 2
    # 오버레이 (RGB)
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = theo_matched  # Red = theory
    overlay[:, :, 1] = skeleton       # Green = extracted
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 0].axis('off')
    
    # 매칭 분석
    kernel = np.ones((5, 5), np.uint8)
    theo_d = cv2.dilate(theo_matched, kernel)
    ext_d = cv2.dilate(skeleton, kernel)
    
    # 매칭된 부분 (노란색으로)
    matched = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    matched[:, :, 0] = np.where(np.logical_and(skeleton > 0, theo_d > 0), 255, 0)  # 추출-이론 매칭
    matched[:, :, 1] = np.where(np.logical_and(skeleton > 0, theo_d > 0), 255, 0)
    # 매칭 안된 부분 표시
    unmatched_ext = np.logical_and(skeleton > 0, theo_d == 0)
    unmatched_theo = np.logical_and(theo_matched > 0, ext_d == 0)
    matched[:, :, 1] = np.where(unmatched_ext, 255, matched[:, :, 1])  # 초록 = 추출만
    matched[:, :, 2] = np.where(unmatched_theo, 255, matched[:, :, 2])  # 파랑 = 이론만
    
    axes[1, 1].imshow(matched)
    axes[1, 1].set_title('Match Analysis\nYellow=matched, Green=ext only, Blue=theo only')
    axes[1, 1].axis('off')
    
    # 결과 텍스트
    result_text = f"""Fill Rate Analysis

Parameters:
  fx = {fx} Hz
  fy = {fy} Hz
  phase = {phase}°

Layout (from extracted):
  Size: {w} x {h} px
  Center: ({cx}, {cy})

Fill Rates:
  Extracted -> Theory: {fill['fill_rate_extracted']*100:.1f}%
  Theory -> Extracted: {fill['fill_rate_theoretical']*100:.1f}%
  Combined: {fill['combined_fill_rate']*100:.1f}%
  IoU: {fill['iou']*100:.1f}%

Pixel Counts:
  Extracted: {fill['extracted_pixels']:,}
  Theoretical: {fill['theoretical_pixels']:,}"""
    
    axes[1, 2].text(0.05, 0.95, result_text, fontsize=10, verticalalignment='top',
                    family='monospace', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Results')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    fig_path = os.path.join(output_dir, f'{base_name}_fillrate.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close()
    
    return fill


if __name__ == '__main__':
    fx, fy, phase = 1350, 1480, 0
    
    results = []
    for img in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img):
            r = analyze_with_layout(img, fx, fy, phase)
            if r:
                results.append((Path(img).name, r))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, r in results:
        print(f"{name}:")
        print(f"  Extracted->Theory: {r['fill_rate_extracted']*100:.1f}%")
        print(f"  Theory->Extracted: {r['fill_rate_theoretical']*100:.1f}%")
        print(f"  Combined: {r['combined_fill_rate']*100:.1f}%")

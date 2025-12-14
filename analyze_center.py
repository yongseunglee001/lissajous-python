#!/usr/bin/env python3
"""
1. Extracted curve에서 중앙 교점 찾기
2. 프레임 정중앙에 배치
3. 0.01초 동안의 이론 궤적 생성 (1350Hz, 1480Hz)
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
    skeleton = (morphology.skeletonize(binary > 0) * 255).astype(np.uint8)
    return binary, skeleton


def find_intersection_points(skeleton):
    """
    골격에서 교점(분기점) 찾기
    교점: 주변 8방향에 3개 이상의 이웃을 가진 점
    """
    h, w = skeleton.shape
    intersections = []
    
    # 각 픽셀 주변의 이웃 수 계산
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    
    # 교점: 골격 픽셀이면서 이웃이 3개 이상
    intersection_mask = np.logical_and(skeleton > 0, neighbor_count >= 3)
    
    points = np.where(intersection_mask)
    for y, x in zip(points[0], points[1]):
        intersections.append((x, y))
    
    return intersections


def find_center_intersection(skeleton):
    """
    가장 중앙에 가까운 교점 찾기
    """
    h, w = skeleton.shape
    center_x, center_y = w // 2, h // 2
    
    intersections = find_intersection_points(skeleton)
    
    if not intersections:
        # 교점이 없으면 골격의 중심점 반환
        points = np.where(skeleton > 0)
        if len(points[0]) > 0:
            return int(np.mean(points[1])), int(np.mean(points[0]))
        return center_x, center_y
    
    # 중앙에 가장 가까운 교점 찾기
    min_dist = float('inf')
    center_intersection = (center_x, center_y)
    
    for x, y in intersections:
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_intersection = (x, y)
    
    return center_intersection


def get_curve_scale(skeleton):
    """추출된 곡선의 스케일(반경) 계산"""
    points = np.where(skeleton > 0)
    if len(points[0]) == 0:
        return 100
    
    center_y = np.mean(points[0])
    center_x = np.mean(points[1])
    
    # 중심에서 각 점까지의 거리의 최대값
    distances = np.sqrt((points[1] - center_x)**2 + (points[0] - center_y)**2)
    return np.max(distances)


def generate_lissajous_01sec(fx, fy, phase_deg, center, scale, img_shape, duration=0.01):
    """
    0.01초 동안의 리사쥬 궤적 생성
    
    Args:
        fx, fy: 주파수 (Hz)
        phase_deg: 위상차 (도)
        center: (x, y) 중심점
        scale: 스케일 (픽셀)
        img_shape: 이미지 크기
        duration: 지속 시간 (초)
    """
    phase = np.radians(phase_deg)
    
    # 0.01초 동안의 샘플
    # 1350Hz면 0.01초에 13.5 사이클
    num_samples = int(max(fx, fy) * duration * 100)  # 충분한 해상도
    t = np.linspace(0, duration, num_samples)
    
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 중심과 스케일 적용
    cx, cy = center
    x_img = (x * scale + cx).astype(np.int32)
    y_img = (y * scale + cy).astype(np.int32)
    
    # 이미지 생성
    h, w = img_shape[:2]
    theo_img = np.zeros((h, w), dtype=np.uint8)
    
    points = np.column_stack((x_img, y_img))
    valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    points = points[valid]
    
    for i in range(len(points) - 1):
        cv2.line(theo_img, tuple(points[i]), tuple(points[i+1]), 255, 2)
    
    return theo_img, len(t), fx * duration, fy * duration


def calculate_fill_rate(extracted, theoretical):
    """채움률 계산"""
    kernel = np.ones((5, 5), np.uint8)
    ext_d = cv2.dilate(extracted, kernel)
    theo_d = cv2.dilate(theoretical, kernel)
    
    ext_pixels = (extracted > 0).sum()
    theo_pixels = (theoretical > 0).sum()
    
    ext_in_theo = np.logical_and(extracted > 0, theo_d > 0).sum()
    theo_in_ext = np.logical_and(theoretical > 0, ext_d > 0).sum()
    
    return {
        'extracted_pixels': int(ext_pixels),
        'theoretical_pixels': int(theo_pixels),
        'extracted_to_theory': ext_in_theo / ext_pixels if ext_pixels > 0 else 0,
        'theory_to_extracted': theo_in_ext / theo_pixels if theo_pixels > 0 else 0,
    }


def analyze_with_center_intersection(image_path, fx=1350, fy=1480, phase=0, 
                                      duration=0.01, output_dir='./results'):
    """중앙 교점 기반 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"Parameters: fx={fx}Hz, fy={fy}Hz, phase={phase}°")
    print(f"Duration: {duration*1000:.1f}ms ({duration}s)")
    print(f"{'='*60}")
    
    # 로드
    original = load_image(image_path)
    if original is None:
        return None
    
    # 디스플레이 검출
    display, bbox = detect_display_region(original)
    disp_h, disp_w = display.shape[:2]
    print(f"Display: {disp_w}x{disp_h}")
    
    # 분석용 리사이즈
    target_size = 500
    scale_factor = target_size / max(disp_h, disp_w)
    resized = cv2.resize(display, (int(disp_w * scale_factor), int(disp_h * scale_factor)))
    res_h, res_w = resized.shape[:2]
    
    # 곡선 추출
    binary, skeleton = extract_curve(resized)
    
    # 중앙 교점 찾기
    center_intersection = find_center_intersection(skeleton)
    print(f"Center intersection found: {center_intersection}")
    
    # 모든 교점 찾기 (시각화용)
    all_intersections = find_intersection_points(skeleton)
    print(f"Total intersection points: {len(all_intersections)}")
    
    # 프레임 정중앙
    frame_center = (res_w // 2, res_h // 2)
    print(f"Frame center: {frame_center}")
    
    # 곡선 스케일 계산
    curve_scale = get_curve_scale(skeleton)
    print(f"Curve scale (radius): {curve_scale:.1f}px")
    
    # 0.01초 이론 궤적 생성 (프레임 정중앙 기준)
    theo_img, num_samples, cycles_x, cycles_y = generate_lissajous_01sec(
        fx, fy, phase, frame_center, curve_scale, skeleton.shape, duration
    )
    print(f"\nTheoretical trajectory ({duration*1000:.1f}ms):")
    print(f"  Samples: {num_samples}")
    print(f"  X cycles: {cycles_x:.2f}")
    print(f"  Y cycles: {cycles_y:.2f}")
    
    # 채움률 계산
    fill = calculate_fill_rate(skeleton, theo_img)
    print(f"\nFill Rate:")
    print(f"  Extracted→Theory: {fill['extracted_to_theory']*100:.1f}%")
    print(f"  Theory→Extracted: {fill['theory_to_extracted']*100:.1f}%")
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cropped Display')
    axes[0, 0].axis('off')
    
    # 골격 + 교점들 표시
    axes[0, 1].imshow(skeleton, cmap='gray')
    if all_intersections:
        xs, ys = zip(*all_intersections)
        axes[0, 1].scatter(xs, ys, c='red', s=20, marker='x', label='Intersections')
    axes[0, 1].scatter(center_intersection[0], center_intersection[1], 
                       c='yellow', s=100, marker='o', edgecolors='red', linewidths=2,
                       label='Center intersection')
    axes[0, 1].scatter(frame_center[0], frame_center[1], 
                       c='cyan', s=100, marker='+', linewidths=2,
                       label='Frame center')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].set_title(f'Extracted Curve\n{len(all_intersections)} intersections')
    axes[0, 1].axis('off')
    
    # 이론 궤적 (0.01초)
    axes[0, 2].imshow(theo_img, cmap='gray')
    axes[0, 2].scatter(frame_center[0], frame_center[1], c='red', s=50, marker='+')
    axes[0, 2].set_title(f'Theory: {duration*1000:.0f}ms trajectory\n{fx}Hz x {fy}Hz, {cycles_x:.1f} x {cycles_y:.1f} cycles')
    axes[0, 2].axis('off')
    
    # Row 2
    # 오버레이
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = theo_img
    overlay[:, :, 1] = skeleton
    axes[1, 0].imshow(overlay)
    axes[1, 0].scatter(frame_center[0], frame_center[1], c='white', s=50, marker='+')
    axes[1, 0].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 0].axis('off')
    
    # 매칭 분석
    kernel = np.ones((5, 5), np.uint8)
    theo_d = cv2.dilate(theo_img, kernel)
    ext_d = cv2.dilate(skeleton, kernel)
    
    matched = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    both = np.logical_and(skeleton > 0, theo_d > 0)
    matched[:, :, 0] = np.where(both, 255, 0)
    matched[:, :, 1] = np.where(both, 255, 0)
    matched[:, :, 1] = np.where(np.logical_and(skeleton > 0, ~both), 255, matched[:, :, 1])
    matched[:, :, 2] = np.where(np.logical_and(theo_img > 0, ext_d == 0), 255, 0)
    
    axes[1, 1].imshow(matched)
    axes[1, 1].set_title('Match: Yellow=both, Green=ext only, Blue=theo only')
    axes[1, 1].axis('off')
    
    # 결과 텍스트
    result_text = f"""Analysis Results

Parameters:
  fx = {fx} Hz
  fy = {fy} Hz
  phase = {phase}°
  duration = {duration*1000:.1f} ms

Trajectory info:
  X cycles: {cycles_x:.2f}
  Y cycles: {cycles_y:.2f}
  Samples: {num_samples}

Center intersection: {center_intersection}
Frame center: {frame_center}
Curve scale: {curve_scale:.1f} px

Fill Rate:
  Extracted→Theory: {fill['extracted_to_theory']*100:.1f}%
  Theory→Extracted: {fill['theory_to_extracted']*100:.1f}%
  
Pixels:
  Extracted: {fill['extracted_pixels']:,}
  Theoretical: {fill['theoretical_pixels']:,}"""
    
    axes[1, 2].text(0.05, 0.95, result_text, fontsize=9, verticalalignment='top',
                    family='monospace', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    fig_path = os.path.join(output_dir, f'{base_name}_center_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close()
    
    return {
        'center_intersection': center_intersection,
        'frame_center': frame_center,
        'fill_rate': fill,
        'cycles': (cycles_x, cycles_y)
    }


if __name__ == '__main__':
    fx, fy, phase = 1350, 1480, 0
    duration = 0.01  # 0.01초
    
    results = []
    for img in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img):
            r = analyze_with_center_intersection(img, fx, fy, phase, duration)
            if r:
                results.append((Path(img).name, r))
    
    print("\n" + "="*60)
    print(f"Summary (0.01s trajectory)")
    print("="*60)
    for name, r in results:
        print(f"\n{name}:")
        print(f"  Center intersection: {r['center_intersection']}")
        print(f"  Cycles in 0.01s: X={r['cycles'][0]:.1f}, Y={r['cycles'][1]:.1f}")
        print(f"  Fill: Ext→Theo {r['fill_rate']['extracted_to_theory']*100:.1f}%, "
              f"Theo→Ext {r['fill_rate']['theory_to_extracted']*100:.1f}%")

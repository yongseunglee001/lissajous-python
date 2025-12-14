#!/usr/bin/env python3
"""
개선된 오실로스코프 리사쥬 패턴 분석 스크립트
오실로스코프 화면에서 리사쥬 곡선을 정확히 추출합니다.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import morphology, measure

# HEIF 지원
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image
from scipy import ndimage


def load_image(image_path):
    """이미지 로드 (HEIF 지원)"""
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


def extract_curve_improved(image):
    """
    개선된 곡선 추출 - 여러 방법 조합
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    results = {}
    
    # 방법 1: 매우 밝은 영역 (선/트레이스)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    results['very_bright'] = (bright_mask, bright_mask.sum() / 255)
    
    # 방법 2: 중간 밝기 이상
    _, mid_bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    results['mid_bright'] = (mid_bright, mid_bright.sum() / 255)
    
    # 방법 3: 적응형 임계값
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, -15)
    results['adaptive'] = (adaptive, adaptive.sum() / 255)
    
    # 방법 4: 녹색/노란색 채널 (오실로스코프 트레이스 색)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    _, lab_thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
    results['lab_bright'] = (lab_thresh, lab_thresh.sum() / 255)
    
    # 방법 5: 색상 기반 (녹색 계열 - 오실로스코프)
    lower_green = np.array([25, 30, 100])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    results['green'] = (green_mask, green_mask.sum() / 255)
    
    # 방법 6: 노란색/녹색 밝은 영역
    lower_lime = np.array([20, 20, 180])
    upper_lime = np.array([100, 255, 255])
    lime_mask = cv2.inRange(hsv, lower_lime, upper_lime)
    results['lime'] = (lime_mask, lime_mask.sum() / 255)
    
    return results


def filter_largest_component(binary_image, min_area_ratio=0.001):
    """가장 큰 연결 성분만 남기고 나머지 제거"""
    h, w = binary_image.shape
    min_area = h * w * min_area_ratio
    
    # 레이블링
    labeled, num_features = ndimage.label(binary_image > 0)
    
    if num_features == 0:
        return binary_image
    
    # 각 레이블의 크기 계산
    sizes = ndimage.sum(binary_image > 0, labeled, range(1, num_features + 1))
    
    # 가장 큰 성분들만 선택 (상위 몇 개)
    sorted_indices = np.argsort(sizes)[::-1]
    
    # 가장 큰 성분(들)만 유지
    result = np.zeros_like(binary_image)
    for i in range(min(3, len(sorted_indices))):  # 상위 3개까지
        if sizes[sorted_indices[i]] >= min_area:
            result[labeled == (sorted_indices[i] + 1)] = 255
    
    return result


def extract_oscilloscope_display(image):
    """
    오실로스코프 디스플레이 영역만 추출
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이미지 중앙 영역에 집중 (디스플레이가 보통 중앙에 있음)
    center_y, center_x = h // 2, w // 2
    
    # 원형 마스크 생성 (오실로스코프 디스플레이가 보통 원형/정사각형)
    radius = min(h, w) // 3
    y, x = np.ogrid[:h, :w]
    center_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.uint8) * 255
    
    return center_mask


def generate_lissajous(fx, fy, phase_deg, size=500, thickness=3):
    """이론적 리사쥬 곡선 이미지 생성"""
    from math import gcd
    
    phase = np.radians(phase_deg)
    
    # 주기 계산
    if fx > 0 and fy > 0:
        g = gcd(int(fx * 1000), int(fy * 1000))
        lcm = (int(fx * 1000) * int(fy * 1000)) // g
        period = lcm / 1000.0 / fx
    else:
        period = 1
    
    t = np.linspace(0, period, 10000)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표
    margin = 0.1
    scale = (1 - margin) * size / 2
    x_img = (x + 1) * scale + margin * size / 2
    y_img = (y + 1) * scale + margin * size / 2
    
    image = np.zeros((size, size), dtype=np.uint8)
    points = np.column_stack((x_img, y_img)).astype(np.int32)
    
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), 255, thickness)
    
    return image


def compare_with_dilation(theo_img, ext_img, dilation=5):
    """팽창을 적용하여 더 관대한 비교"""
    size = max(theo_img.shape[0], ext_img.shape[0], theo_img.shape[1], ext_img.shape[1])
    
    theo_resized = cv2.resize(theo_img, (size, size))
    ext_resized = cv2.resize(ext_img, (size, size))
    
    # 팽창 적용 (약간의 오차 허용)
    kernel = np.ones((dilation, dilation), np.uint8)
    theo_dilated = cv2.dilate(theo_resized, kernel, iterations=1)
    ext_dilated = cv2.dilate(ext_resized, kernel, iterations=1)
    
    theo_bin = (theo_resized > 127).astype(np.uint8)
    ext_bin = (ext_resized > 127).astype(np.uint8)
    theo_dil = (theo_dilated > 127).astype(np.uint8)
    ext_dil = (ext_dilated > 127).astype(np.uint8)
    
    # 표준 메트릭
    intersection = np.logical_and(theo_bin, ext_bin).sum()
    union = np.logical_or(theo_bin, ext_bin).sum()
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (theo_bin.sum() + ext_bin.sum()) if (theo_bin.sum() + ext_bin.sum()) > 0 else 0
    
    # 커버리지 메트릭 (팽창 적용)
    theo_covered = np.logical_and(theo_bin, ext_dil).sum() / theo_bin.sum() if theo_bin.sum() > 0 else 0
    ext_covered = np.logical_and(ext_bin, theo_dil).sum() / ext_bin.sum() if ext_bin.sum() > 0 else 0
    
    # 거리 기반 점수
    from scipy.ndimage import distance_transform_edt
    dist_from_theo = distance_transform_edt(1 - theo_bin)
    dist_from_ext = distance_transform_edt(1 - ext_bin)
    
    # 추출된 픽셀들의 이론 곡선까지 평균 거리
    if ext_bin.sum() > 0:
        mean_dist_to_theo = dist_from_theo[ext_bin > 0].mean()
    else:
        mean_dist_to_theo = float('inf')
    
    # 이론 픽셀들의 추출 곡선까지 평균 거리
    if theo_bin.sum() > 0:
        mean_dist_to_ext = dist_from_ext[theo_bin > 0].mean()
    else:
        mean_dist_to_ext = float('inf')
    
    return {
        'iou': iou,
        'dice': dice,
        'theo_coverage': theo_covered,
        'ext_coverage': ext_covered,
        'mean_dist_to_theo': mean_dist_to_theo,
        'mean_dist_to_ext': mean_dist_to_ext,
        'combined_score': (iou + dice + theo_covered + ext_covered) / 4  # 종합 점수
    }


def find_best_match_improved(extracted_binary, freq_pairs=None, phase_range=(0, 180, 10)):
    """개선된 최적 파라미터 탐색"""
    if freq_pairs is None:
        freq_pairs = [
            (1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
            (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2),
            (3, 5), (5, 3), (4, 5), (5, 4), (1, 4), (4, 1),
            (5, 6), (6, 5), (5, 7), (7, 5)
        ]
    
    phases = np.arange(phase_range[0], phase_range[1] + 1, phase_range[2])
    size = max(extracted_binary.shape)
    
    # 추출 이미지 정제
    cleaned = filter_largest_component(extracted_binary)
    
    # 골격화
    skeleton = morphology.skeletonize(cleaned > 0).astype(np.uint8) * 255
    
    best_score = -1
    best_params = None
    all_results = []
    
    print(f"  탐색 중: {len(freq_pairs)} 주파수 쌍 x {len(phases)} 위상 = {len(freq_pairs) * len(phases)} 조합")
    
    for fx, fy in freq_pairs:
        for phase in phases:
            theo_img = generate_lissajous(fx, fy, phase, size=size, thickness=5)
            metrics = compare_with_dilation(theo_img, skeleton, dilation=10)
            
            score = metrics['combined_score']
            all_results.append({
                'fx': fx, 'fy': fy, 'phase': phase,
                'score': score, **metrics
            })
            
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    # 상위 결과들 출력
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
    print("\n  상위 5개 결과:")
    for r in sorted_results:
        print(f"    fx={r['fx']}, fy={r['fy']}, φ={r['phase']}° -> 점수: {r['score']:.4f}")
    
    return best_params, best_score, skeleton, all_results


def analyze_image_improved(image_path, output_dir='./results'):
    """개선된 분석 파이프라인"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"분석: {image_path}")
    print(f"{'='*70}")
    
    image = load_image(image_path)
    if image is None:
        print("Error: 이미지를 로드할 수 없습니다")
        return None
    
    h, w = image.shape[:2]
    print(f"크기: {w}x{h}")
    
    # 다양한 추출 방법 시도
    extractions = extract_curve_improved(image)
    
    print("\n추출 결과:")
    for method, (mask, pixel_count) in extractions.items():
        print(f"  {method}: {int(pixel_count):,} 픽셀")
    
    # 가장 적절한 추출 방법 선택
    # (너무 많지도, 너무 적지도 않은 픽셀 수)
    target_ratio = 0.01  # 이미지의 1% 정도
    target_pixels = h * w * target_ratio
    
    best_method = None
    best_diff = float('inf')
    
    for method, (mask, pixel_count) in extractions.items():
        if pixel_count > 100:  # 최소한의 픽셀이 있어야 함
            diff = abs(pixel_count - target_pixels)
            if diff < best_diff:
                best_diff = diff
                best_method = method
    
    if best_method is None:
        # 가장 많은 픽셀을 가진 방법 사용
        best_method = max(extractions.keys(), key=lambda x: extractions[x][1])
    
    selected_mask = extractions[best_method][0]
    print(f"\n선택된 방법: {best_method}")
    
    # 가장 큰 연결 성분만 유지
    cleaned = filter_largest_component(selected_mask, min_area_ratio=0.0001)
    
    # 최적 파라미터 탐색
    print("\n최적 파라미터 탐색...")
    best_params, best_score, skeleton, all_results = find_best_match_improved(cleaned)
    
    fx, fy, phase = best_params
    print(f"\n최적 파라미터: fx={fx}, fy={fy}, φ={phase}°")
    print(f"종합 점수: {best_score:.4f}")
    
    # 시각화
    size = max(h, w)
    theo_img = generate_lissajous(fx, fy, phase, size=size, thickness=5)
    metrics = compare_with_dilation(theo_img, skeleton, dilation=10)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(selected_mask, cmap='gray')
    axes[0, 1].set_title(f'Extracted ({best_method})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cleaned, cmap='gray')
    axes[0, 2].set_title('Cleaned (largest component)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(skeleton, cmap='gray')
    axes[0, 3].set_title('Skeleton')
    axes[0, 3].axis('off')
    
    # Row 2
    axes[1, 0].imshow(theo_img, cmap='gray')
    axes[1, 0].set_title(f'Theory: fx={fx}, fy={fy}, phase={phase}deg')
    axes[1, 0].axis('off')
    
    # 오버레이
    skeleton_resized = cv2.resize(skeleton, (size, size))
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    overlay[:, :, 2] = theo_img
    overlay[:, :, 1] = skeleton_resized
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 1].axis('off')
    
    # 차이 이미지
    diff = np.abs(theo_img.astype(float) - skeleton_resized.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference')
    axes[1, 2].axis('off')
    
    # 메트릭
    metrics_text = f"""Comparison Metrics:

IoU: {metrics['iou']:.4f}
Dice: {metrics['dice']:.4f}
Theory Coverage: {metrics['theo_coverage']:.4f}
Extracted Coverage: {metrics['ext_coverage']:.4f}
Mean Dist to Theory: {metrics['mean_dist_to_theo']:.2f}px
Mean Dist to Extracted: {metrics['mean_dist_to_ext']:.2f}px

Combined Score: {metrics['combined_score']:.4f}

Best Parameters:
  fx = {fx}
  fy = {fy}
  phase = {phase} deg"""
    
    axes[1, 3].text(0.05, 0.95, metrics_text, fontsize=11,
                    verticalalignment='top', family='monospace',
                    transform=axes[1, 3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 3].set_title('Results')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    result_path = os.path.join(output_dir, f'{base_name}_analysis_v2.png')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"\n결과 저장: {result_path}")
    plt.close()
    
    return {
        'params': best_params,
        'metrics': metrics,
        'all_results': all_results
    }


if __name__ == '__main__':
    for img in ['results/IMG_0361.png', 'results/IMG_0383.png']:
        if os.path.exists(img):
            analyze_image_improved(img)

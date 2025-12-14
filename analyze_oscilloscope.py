#!/usr/bin/env python3
"""
오실로스코프 리사쥬 패턴 분석 스크립트
녹색 화면에서 곡선을 추출하고 이론 곡선과 비교합니다.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# HEIF 지원
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    pass


def load_image(image_path):
    """이미지 로드 (HEIF 지원)"""
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(image_path)


def extract_oscilloscope_curve(image, method='green'):
    """
    오실로스코프 화면에서 곡선을 추출합니다.
    
    Args:
        image: BGR 이미지
        method: 'green' (녹색기반), 'bright' (밝기기반), 'edge' (에지기반)
    
    Returns:
        이진 마스크와 골격 이미지
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'green':
        # 녹색 영역 추출
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
    elif method == 'bright':
        # 밝은 영역 추출
        _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    else:
        # 에지 검출
        mask = cv2.Canny(gray, 50, 150)
    
    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 골격화
    from skimage import morphology
    skeleton = morphology.skeletonize(mask > 0).astype(np.uint8) * 255
    
    return mask, skeleton


def generate_lissajous(fx, fy, phase_deg, size=500, thickness=2):
    """이론적 리사쥬 곡선 이미지 생성"""
    from math import gcd
    
    phase = np.radians(phase_deg)
    
    # 주기 계산
    g = gcd(int(fx * 1000), int(fy * 1000))
    lcm = (int(fx * 1000) * int(fy * 1000)) // g
    period = lcm / 1000.0 / fx
    
    t = np.linspace(0, period, 10000)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표로 변환
    margin = 0.1
    scale = (1 - margin) * size / 2
    x_img = (x + 1) * scale + margin * size / 2
    y_img = (y + 1) * scale + margin * size / 2
    
    # 이미지 생성
    image = np.zeros((size, size), dtype=np.uint8)
    points = np.column_stack((x_img, y_img)).astype(np.int32)
    
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), 255, thickness)
    
    return image


def compare_images(theo_img, ext_img):
    """두 이진 이미지 비교"""
    # 크기 맞추기
    size = max(theo_img.shape[0], ext_img.shape[0], theo_img.shape[1], ext_img.shape[1])
    theo_resized = cv2.resize(theo_img, (size, size))
    ext_resized = cv2.resize(ext_img, (size, size))
    
    theo_bin = (theo_resized > 127).astype(np.uint8)
    ext_bin = (ext_resized > 127).astype(np.uint8)
    
    intersection = np.logical_and(theo_bin, ext_bin).sum()
    union = np.logical_or(theo_bin, ext_bin).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (theo_bin.sum() + ext_bin.sum()) if (theo_bin.sum() + ext_bin.sum()) > 0 else 0
    
    return {
        'iou': iou,
        'dice': dice,
        'theo_pixels': theo_bin.sum(),
        'ext_pixels': ext_bin.sum(),
        'intersection': intersection
    }


def find_best_match(extracted_skeleton, freq_pairs=None, phase_range=(0, 180, 5)):
    """최적 파라미터 탐색"""
    if freq_pairs is None:
        freq_pairs = [
            (1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
            (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2),
            (3, 5), (5, 3), (4, 5), (5, 4), (1, 4), (4, 1)
        ]
    
    phases = np.arange(phase_range[0], phase_range[1] + 1, phase_range[2])
    size = max(extracted_skeleton.shape)
    
    best_score = -1
    best_params = None
    results = []
    
    for fx, fy in freq_pairs:
        for phase in phases:
            theo_img = generate_lissajous(fx, fy, phase, size=size, thickness=3)
            metrics = compare_images(theo_img, extracted_skeleton)
            
            score = (metrics['iou'] + metrics['dice']) / 2
            results.append({
                'fx': fx, 'fy': fy, 'phase': phase,
                'score': score, 'iou': metrics['iou'], 'dice': metrics['dice']
            })
            
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    return best_params, best_score, results


def analyze_image(image_path, output_dir='./results'):
    """이미지 분석 전체 파이프라인"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"분석 중: {image_path}")
    print(f"{'='*60}")
    
    # 이미지 로드
    image = load_image(image_path)
    if image is None:
        print(f"Error: 이미지를 로드할 수 없습니다")
        return None
    
    print(f"이미지 크기: {image.shape}")
    
    # 녹색 곡선 추출 시도
    methods = ['green', 'bright', 'edge']
    best_extraction = None
    best_method = None
    max_pixels = 0
    
    for method in methods:
        mask, skeleton = extract_oscilloscope_curve(image, method)
        pixel_count = skeleton.sum() / 255
        print(f"  {method}: {int(pixel_count)} 픽셀")
        
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            best_extraction = (mask, skeleton)
            best_method = method
    
    print(f"\n사용 방법: {best_method} ({int(max_pixels)} 픽셀)")
    
    mask, skeleton = best_extraction
    
    # 최적 파라미터 탐색
    print("\n최적 파라미터 탐색 중...")
    best_params, best_score, all_results = find_best_match(skeleton)
    
    print(f"\n최적 파라미터:")
    print(f"  fx = {best_params[0]}, fy = {best_params[1]}")
    print(f"  위상차 = {best_params[2]}°")
    print(f"  점수 = {best_score:.4f}")
    
    # 시각화
    size = max(image.shape[:2])
    theo_img = generate_lissajous(best_params[0], best_params[1], best_params[2], 
                                   size=size, thickness=3)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('원본 이미지')
    axes[0, 0].axis('off')
    
    # 추출된 마스크
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title(f'추출 마스크 ({best_method})')
    axes[0, 1].axis('off')
    
    # 골격화
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('골격화된 곡선')
    axes[0, 2].axis('off')
    
    # 이론 곡선
    axes[1, 0].imshow(theo_img, cmap='gray')
    axes[1, 0].set_title(f'이론 곡선\nfx={best_params[0]}, fy={best_params[1]}, φ={best_params[2]}°')
    axes[1, 0].axis('off')
    
    # 오버레이
    skeleton_resized = cv2.resize(skeleton, (size, size))
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    overlay[:, :, 2] = theo_img      # Red = 이론
    overlay[:, :, 1] = skeleton_resized  # Green = 추출
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('오버레이 (빨강=이론, 초록=추출)')
    axes[1, 1].axis('off')
    
    # 메트릭
    metrics = compare_images(theo_img, skeleton_resized)
    metrics_text = (
        f"비교 메트릭:\n\n"
        f"  IoU: {metrics['iou']:.4f}\n"
        f"  Dice: {metrics['dice']:.4f}\n\n"
        f"최적 파라미터:\n\n"
        f"  fx = {best_params[0]}\n"
        f"  fy = {best_params[1]}\n"
        f"  위상차 = {best_params[2]}°"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_title('분석 결과')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 저장
    base_name = Path(image_path).stem
    result_path = os.path.join(output_dir, f'{base_name}_analysis.png')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"\n결과 저장: {result_path}")
    
    plt.close()
    
    return {
        'params': best_params,
        'score': best_score,
        'metrics': metrics
    }


if __name__ == '__main__':
    # 변환된 PNG 파일들 분석
    results_dir = './results'
    
    images = [
        'results/IMG_0361.png',
        'results/IMG_0383.png'
    ]
    
    for img in images:
        if os.path.exists(img):
            result = analyze_image(img, results_dir)
            print("\n")

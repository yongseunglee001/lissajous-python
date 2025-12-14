#!/usr/bin/env python3
"""
최적화된 오실로스코프 리사쥬 패턴 분석 스크립트
이미지를 축소하여 빠르게 처리합니다.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import morphology

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image


# 처리할 이미지 크기 (성능을 위해 축소)
TARGET_SIZE = 800


def load_and_resize(image_path, target_size=TARGET_SIZE):
    """이미지 로드 및 리사이즈"""
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            rgb = np.array(img)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    return resized


def extract_bright_curve(image):
    """밝은 곡선 추출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 임계값
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -10)
    
    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def get_largest_contour(binary):
    """가장 큰 컨투어를 추출하여 마스크 생성"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary
    
    # 가장 큰 컨투어
    largest = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    cv2.drawContours(mask, [largest], -1, 255, 3)
    
    return mask


def generate_lissajous(fx, fy, phase_deg, size=TARGET_SIZE, thickness=2):
    """이론적 리사쥬 곡선"""
    from math import gcd
    
    phase = np.radians(phase_deg)
    
    if fx > 0 and fy > 0:
        g = gcd(int(fx * 1000), int(fy * 1000))
        lcm = (int(fx * 1000) * int(fy * 1000)) // g
        period = lcm / 1000.0 / fx
    else:
        period = 1
    
    t = np.linspace(0, period, 5000)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    margin = 0.1
    scale = (1 - margin) * size / 2
    x_img = (x + 1) * scale + margin * size / 2
    y_img = (y + 1) * scale + margin * size / 2
    
    image = np.zeros((size, size), dtype=np.uint8)
    points = np.column_stack((x_img, y_img)).astype(np.int32)
    
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), 255, thickness)
    
    return image


def quick_compare(theo, extracted):
    """빠른 비교 (IoU만)"""
    # 크기 맞춤
    size = max(theo.shape[0], extracted.shape[0])
    theo_r = cv2.resize(theo, (size, size))
    ext_r = cv2.resize(extracted, (size, size))
    
    # 팽창으로 관대하게
    kernel = np.ones((5, 5), np.uint8)
    theo_d = cv2.dilate(theo_r, kernel)
    ext_d = cv2.dilate(ext_r, kernel)
    
    # IoU with dilation
    intersection = np.logical_and(theo_d > 0, ext_d > 0).sum()
    union = np.logical_or(theo_d > 0, ext_d > 0).sum()
    
    return intersection / union if union > 0 else 0


def find_best_params(extracted, use_fine=True):
    """최적 파라미터 탐색"""
    freq_pairs = [
        (1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
        (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2),
        (3, 5), (5, 3), (1, 4), (4, 1)
    ]
    
    # 1차: 거칠게
    phases = range(0, 180, 15)
    size = max(extracted.shape)
    
    best_score = -1
    best_params = (1, 1, 0)
    
    for fx, fy in freq_pairs:
        for phase in phases:
            theo = generate_lissajous(fx, fy, phase, size=size)
            score = quick_compare(theo, extracted)
            
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    # 2차: 최고 주파수 쌍에서 위상 세밀 탐색
    if use_fine:
        fx, fy, rough_phase = best_params
        fine_phases = range(max(0, rough_phase - 15), min(181, rough_phase + 16), 3)
        
        for phase in fine_phases:
            theo = generate_lissajous(fx, fy, phase, size=size)
            score = quick_compare(theo, extracted)
            
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    return best_params, best_score


def analyze(image_path, output_dir='./results'):
    """분석 실행"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # 로드 및 리사이즈
    image = load_and_resize(image_path)
    if image is None:
        print("Failed to load image")
        return None
    
    h, w = image.shape[:2]
    print(f"Working size: {w}x{h}")
    
    # 곡선 추출
    binary = extract_bright_curve(image)
    cleaned = get_largest_contour(binary)
    skeleton = (morphology.skeletonize(cleaned > 0) * 255).astype(np.uint8)
    
    # 최적 파라미터 탐색
    print("Finding best parameters...")
    (fx, fy, phase), score = find_best_params(skeleton)
    
    print(f"\nBest match: fx={fx}, fy={fy}, phase={phase}deg")
    print(f"Score: {score:.4f}")
    
    # 시각화
    size = max(h, w)
    theo = generate_lissajous(fx, fy, phase, size=size)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (resized)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary extraction')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Skeleton')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(theo, cmap='gray')
    axes[1, 0].set_title(f'Theory: fx={fx}, fy={fy}, phase={phase}deg')
    axes[1, 0].axis('off')
    
    # 오버레이
    skeleton_r = cv2.resize(skeleton, (size, size))
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    overlay[:, :, 2] = theo
    overlay[:, :, 1] = skeleton_r
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 1].axis('off')
    
    # 결과 텍스트
    text = f"""Analysis Results
    
Parameters:
  fx = {fx}
  fy = {fy}
  phase = {phase} deg
  
Score (IoU): {score:.4f}

Ratio: {fx}:{fy}
"""
    axes[1, 2].text(0.1, 0.5, text, fontsize=14, verticalalignment='center',
                    family='monospace', 
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    result_path = os.path.join(output_dir, f'{base_name}_result.png')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {result_path}")
    plt.close()
    
    return {'fx': fx, 'fy': fy, 'phase': phase, 'score': score}


if __name__ == '__main__':
    results = []
    for img in ['results/IMG_0361.png', 'results/IMG_0383.png']:
        if os.path.exists(img):
            r = analyze(img)
            if r:
                results.append((img, r))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for img, r in results:
        print(f"{Path(img).name}: fx={r['fx']}, fy={r['fy']}, phase={r['phase']}deg, score={r['score']:.4f}")

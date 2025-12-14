#!/usr/bin/env python3
"""
정확한 파라미터로 리사쥬 곡선 비교
사용자 지정: fx=1350Hz, fy=1480Hz, phase=0°
"""

import os
import numpy as np
import cv2
from pathlib import Path
from math import gcd

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
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
                margin = int(min(bw, bh) * 0.02)
                x, y = max(0, x - margin), max(0, y - margin)
                bw = min(w - x, bw + 2 * margin)
                bh = min(h - y, bh + 2 * margin)
                return image[y:y+bh, x:x+bw], (x, y, bw, bh)
    
    margin = int(min(h, w) * 0.1)
    return image[margin:h-margin, margin:w-margin], (margin, margin, w-2*margin, h-2*margin)


def generate_lissajous(fx, fy, phase_deg, num_points=20000):
    """
    리사쥬 곡선 생성
    실제 주파수(Hz) 사용
    """
    phase = np.radians(phase_deg)
    
    # 주파수 비율 단순화
    g = gcd(int(fx), int(fy))
    ratio_x = fx // g
    ratio_y = fy // g
    
    print(f"  Frequency ratio: {ratio_x}:{ratio_y} (simplified from {fx}:{fy})")
    
    # 완전한 패턴을 위한 시간 범위
    # LCM / fx 주기가 필요
    lcm_freq = (fx * fy) // g
    period = lcm_freq / fx / fy  # 초 단위
    
    # 더 많은 주기를 위해 확장
    t = np.linspace(0, 1.0, num_points)  # 1초간
    
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    return x, y, ratio_x, ratio_y


def draw_lissajous(image, fx, fy, phase, color=(0, 255, 255), thickness=2):
    """이미지에 리사쥬 곡선 그리기"""
    h, w = image.shape[:2]
    result = image.copy()
    
    x, y, _, _ = generate_lissajous(fx, fy, phase)
    
    center_x, center_y = w // 2, h // 2
    scale = min(w, h) * 0.45
    
    x_img = (x * scale + center_x).astype(np.int32)
    y_img = (y * scale + center_y).astype(np.int32)
    
    points = np.column_stack((x_img, y_img))
    for i in range(len(points) - 1):
        cv2.line(result, tuple(points[i]), tuple(points[i+1]), color, thickness)
    
    return result


def extract_curve(image):
    """곡선 추출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -5)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def calculate_iou(theo_x, theo_y, extracted_binary):
    """IoU 계산"""
    h, w = extracted_binary.shape
    
    # 이론 곡선 이미지 생성
    theo_img = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    scale = min(w, h) * 0.45
    
    x_img = (theo_x * scale + center_x).astype(np.int32)
    y_img = (theo_y * scale + center_y).astype(np.int32)
    
    points = np.column_stack((x_img, y_img))
    for i in range(len(points) - 1):
        cv2.line(theo_img, tuple(points[i]), tuple(points[i+1]), 255, 3)
    
    # 팽창
    kernel = np.ones((7, 7), np.uint8)
    theo_d = cv2.dilate(theo_img, kernel)
    ext_d = cv2.dilate(extracted_binary, kernel)
    
    intersection = np.logical_and(theo_d > 0, ext_d > 0).sum()
    union = np.logical_or(theo_d > 0, ext_d > 0).sum()
    
    return intersection / union if union > 0 else 0, theo_img


def analyze_with_params(image_path, fx, fy, phase, output_dir='./results'):
    """지정된 파라미터로 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"Parameters: fx={fx}Hz, fy={fy}Hz, phase={phase}°")
    print(f"{'='*60}")
    
    # 이미지 로드
    original = load_image(image_path)
    if original is None:
        print("Failed to load")
        return None
    
    orig_h, orig_w = original.shape[:2]
    print(f"Original: {orig_w}x{orig_h}")
    
    # 디스플레이 영역 검출
    display, bbox = detect_display_region(original)
    disp_h, disp_w = display.shape[:2]
    print(f"Display: {disp_w}x{disp_h}")
    
    # 리사쥬 곡선 생성
    theo_x, theo_y, ratio_x, ratio_y = generate_lissajous(fx, fy, phase)
    
    # 분석용 리사이즈
    target_size = 500
    scale = target_size / max(disp_h, disp_w)
    resized = cv2.resize(display, (int(disp_w * scale), int(disp_h * scale)))
    
    # 곡선 추출
    extracted = extract_curve(resized)
    
    # IoU 계산
    iou, theo_img = calculate_iou(theo_x, theo_y, extracted)
    print(f"IoU Score: {iou:.4f}")
    
    # 디스플레이에 오버레이
    overlay_display = draw_lissajous(
        display, fx, fy, phase,
        color=(0, 255, 255),
        thickness=max(2, disp_w // 200)
    )
    
    # 원본에 오버레이
    overlay_full = original.copy()
    x, y, w, h = bbox
    overlay_full[y:y+h, x:x+w] = overlay_display
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                          fill=False, edgecolor='yellow', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original + Display Region')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Cropped Display')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(extracted, cmap='gray')
    axes[0, 2].set_title('Extracted Curve')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(theo_img, cmap='gray')
    axes[1, 0].set_title(f'Theoretical: {fx}:{fy}Hz ({ratio_x}:{ratio_y}), phase={phase}deg')
    axes[1, 0].axis('off')
    
    # 오버레이
    overlay_rgb = np.zeros((*extracted.shape, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = theo_img
    overlay_rgb[:, :, 1] = extracted
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title(f'Overlay (R=theory, G=extracted)\nIoU={iou:.4f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Final: Yellow = Theoretical Lissajous')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    base_name = Path(image_path).stem
    fig_path = os.path.join(output_dir, f'{base_name}_compare.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
    cv2.imwrite(overlay_path, overlay_full)
    print(f"Saved: {overlay_path}")
    
    return {'iou': iou, 'ratio': f"{ratio_x}:{ratio_y}"}


if __name__ == '__main__':
    # 사용자 지정 파라미터
    fx = 1350  # Hz
    fy = 1480  # Hz
    phase = 0  # degrees
    
    results = []
    for img_path in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img_path):
            r = analyze_with_params(img_path, fx, fy, phase)
            if r:
                results.append((Path(img_path).name, r))
    
    print("\n" + "="*60)
    print(f"Summary (fx={fx}Hz, fy={fy}Hz, phase={phase}°)")
    print("="*60)
    for name, r in results:
        print(f"{name}: Ratio={r['ratio']}, IoU={r['iou']:.4f}")

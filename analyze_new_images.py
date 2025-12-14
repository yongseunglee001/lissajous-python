#!/usr/bin/env python3
"""
오실로스코프 리사쥬 패턴 분석
1. 이미지에서 디스플레이(직사각형) 영역 검출
2. 해당 영역에서 리사쥬 곡선 추출
3. 이론 곡선과 비교 및 오버레이
"""

import os
import numpy as np
import cv2
from pathlib import Path
from skimage import morphology
import matplotlib.pyplot as plt

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image


def load_image(image_path):
    """이미지 로드"""
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


def detect_display_region(image, debug=False):
    """
    이미지에서 오실로스코프 디스플레이(직사각형) 영역을 검출합니다.
    
    Returns:
        cropped: 크롭된 디스플레이 영역
        bbox: (x, y, w, h) 바운딩 박스
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 방법 1: 밝은 영역 검출 (디스플레이가 보통 밝음)
    # 여러 임계값 시도
    for thresh_val in [50, 80, 100, 120]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # 가장 큰 사각형 컨투어 찾기
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            
            # 이미지의 1% ~ 90% 크기의 영역만 고려
            if area < h * w * 0.01 or area > h * w * 0.9:
                continue
            
            # 사각형 근사
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # 바운딩 박스
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # 가로세로 비율 체크 (너무 길쭉하지 않은 것)
            aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 99
            if aspect < 3:  # 적당한 비율
                # 약간의 여백 추가
                margin = int(min(bw, bh) * 0.02)
                x = max(0, x - margin)
                y = max(0, y - margin)
                bw = min(w - x, bw + 2 * margin)
                bh = min(h - y, bh + 2 * margin)
                
                cropped = image[y:y+bh, x:x+bw]
                
                if debug:
                    print(f"Found display: x={x}, y={y}, w={bw}, h={bh}, thresh={thresh_val}")
                
                return cropped, (x, y, bw, bh)
    
    # 실패시 중앙 영역 반환
    margin = int(min(h, w) * 0.1)
    center_crop = image[margin:h-margin, margin:w-margin]
    return center_crop, (margin, margin, w-2*margin, h-2*margin)


def extract_curve_from_display(display_image):
    """디스플레이 이미지에서 곡선을 추출"""
    gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -5)
    
    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 골격화
    skeleton = (morphology.skeletonize(binary > 0) * 255).astype(np.uint8)
    
    return binary, skeleton


def generate_lissajous_points(fx, fy, phase_deg, num_points=10000):
    """리사쥬 좌표 생성"""
    from math import gcd
    phase = np.radians(phase_deg)
    
    g = gcd(int(fx * 1000), int(fy * 1000))
    lcm = (int(fx * 1000) * int(fy * 1000)) // g
    period = lcm / 1000.0 / fx
    
    t = np.linspace(0, period, num_points)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    return x, y


def draw_lissajous_on_display(display_image, fx, fy, phase, color=(0, 255, 255), thickness=2, margin=0.05):
    """디스플레이 이미지 위에 리사쥬 곡선 그리기"""
    h, w = display_image.shape[:2]
    result = display_image.copy()
    
    x, y = generate_lissajous_points(fx, fy, phase)
    
    center_x, center_y = w // 2, h // 2
    scale = min(w, h) * (1 - margin) / 2
    
    x_img = (x * scale + center_x).astype(np.int32)
    y_img = (y * scale + center_y).astype(np.int32)
    
    points = np.column_stack((x_img, y_img))
    for i in range(len(points) - 1):
        cv2.line(result, tuple(points[i]), tuple(points[i+1]), color, thickness)
    
    return result


def quick_iou(theo_x, theo_y, extracted, margin=0.05):
    """빠른 IoU 계산"""
    h, w = extracted.shape[:2]
    size = max(h, w)
    
    center_x, center_y = w // 2, h // 2
    scale = min(w, h) * (1 - margin) / 2
    
    theo_img = np.zeros((h, w), dtype=np.uint8)
    x_img = (theo_x * scale + center_x).astype(np.int32)
    y_img = (theo_y * scale + center_y).astype(np.int32)
    
    points = np.column_stack((x_img, y_img))
    for i in range(len(points) - 1):
        cv2.line(theo_img, tuple(points[i]), tuple(points[i+1]), 255, 3)
    
    # 팽창
    kernel = np.ones((7, 7), np.uint8)
    theo_d = cv2.dilate(theo_img, kernel)
    ext_d = cv2.dilate(extracted, kernel)
    
    intersection = np.logical_and(theo_d > 0, ext_d > 0).sum()
    union = np.logical_or(theo_d > 0, ext_d > 0).sum()
    
    return intersection / union if union > 0 else 0


def find_best_params(extracted):
    """최적 파라미터 탐색"""
    freq_pairs = [
        (1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
        (2, 3), (3, 2), (3, 4), (4, 3), (1, 4), (4, 1)
    ]
    
    best_score = -1
    best_params = (1, 1, 0)
    
    # 1차: 거친 탐색
    for fx, fy in freq_pairs:
        for phase in range(0, 180, 15):
            x, y = generate_lissajous_points(fx, fy, phase)
            score = quick_iou(x, y, extracted)
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    # 2차: 세밀 탐색
    fx, fy, rough_phase = best_params
    for phase in range(max(0, rough_phase - 20), min(180, rough_phase + 21), 2):
        x, y = generate_lissajous_points(fx, fy, phase)
        score = quick_iou(x, y, extracted)
        if score > best_score:
            best_score = score
            best_params = (fx, fy, phase)
    
    return best_params, best_score


def analyze_image(image_path, output_dir='./results'):
    """이미지 분석 전체 파이프라인"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # 1. 이미지 로드
    original = load_image(image_path)
    if original is None:
        print("Failed to load image")
        return None
    
    orig_h, orig_w = original.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")
    
    # 2. 디스플레이 영역 검출 및 크롭
    print("Step 1: Detecting display region...")
    display, bbox = detect_display_region(original, debug=True)
    disp_h, disp_w = display.shape[:2]
    print(f"Display region: {disp_w}x{disp_h}")
    
    # 3. 분석용 리사이즈 (속도용)
    target_size = 600
    scale = target_size / max(disp_h, disp_w)
    resized = cv2.resize(display, (int(disp_w * scale), int(disp_h * scale)))
    
    # 4. 곡선 추출
    print("Step 2: Extracting curve...")
    binary, skeleton = extract_curve_from_display(resized)
    
    # 5. 최적 파라미터 탐색
    print("Step 3: Finding best parameters...")
    (fx, fy, phase), score = find_best_params(skeleton)
    print(f"Best match: fx={fx}, fy={fy}, phase={phase}°, score={score:.4f}")
    
    # 6. 결과 시각화
    base_name = Path(image_path).stem
    
    # 디스플레이에 오버레이
    overlay_display = draw_lissajous_on_display(
        display, fx, fy, phase,
        color=(0, 255, 255),  # Yellow
        thickness=max(2, disp_w // 200)
    )
    
    # 원본에 오버레이 (디스플레이 위치에)
    overlay_full = original.copy()
    x, y, w, h = bbox
    overlay_full[y:y+h, x:x+w] = overlay_display
    
    # 시각화 플롯
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: 추출 과정
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    # 디스플레이 영역 표시
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                          fill=False, edgecolor='yellow', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original (display region in yellow)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Cropped Display ({disp_w}x{disp_h})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Extracted Curve (skeleton)')
    axes[0, 2].axis('off')
    
    # Row 2: 비교 결과
    theo_img = np.zeros_like(skeleton)
    tx, ty = generate_lissajous_points(fx, fy, phase)
    h_sk, w_sk = skeleton.shape
    center_x, center_y = w_sk // 2, h_sk // 2
    scale_draw = min(w_sk, h_sk) * 0.95 / 2
    x_img = (tx * scale_draw + center_x).astype(np.int32)
    y_img = (ty * scale_draw + center_y).astype(np.int32)
    points = np.column_stack((x_img, y_img))
    for i in range(len(points) - 1):
        cv2.line(theo_img, tuple(points[i]), tuple(points[i+1]), 255, 2)
    
    axes[1, 0].imshow(theo_img, cmap='gray')
    axes[1, 0].set_title(f'Theoretical: {fx}:{fy}, {phase}deg')
    axes[1, 0].axis('off')
    
    # 오버레이 (RGB로 표시: 빨강=이론, 초록=추출)
    overlay_rgb = np.zeros((h_sk, w_sk, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = theo_img
    overlay_rgb[:, :, 1] = skeleton
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Overlay (R=theory, G=extracted)')
    axes[1, 1].axis('off')
    
    # 최종 결과
    axes[1, 2].imshow(cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Result: {fx}:{fy} ratio, phase={phase}deg\nScore={score:.4f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 저장
    fig_path = os.path.join(output_dir, f'{base_name}_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis: {fig_path}")
    plt.close()
    
    # 오버레이 이미지 저장
    overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
    cv2.imwrite(overlay_path, overlay_full)
    print(f"Saved overlay: {overlay_path}")
    
    return {
        'fx': fx, 'fy': fy, 'phase': phase, 'score': score,
        'display_size': (disp_w, disp_h)
    }


if __name__ == '__main__':
    results = []
    
    for img_path in ['results/IMG_0350.png', 'results/IMG_0355.png']:
        if os.path.exists(img_path):
            r = analyze_image(img_path)
            if r:
                results.append((Path(img_path).name, r))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, r in results:
        print(f"{name}:")
        print(f"  Ratio: {r['fx']}:{r['fy']}")
        print(f"  Phase: {r['phase']}°")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Display: {r['display_size'][0]}x{r['display_size'][1]}")

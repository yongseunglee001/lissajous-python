#!/usr/bin/env python3
"""
원본 이미지 위에 예측된 리사쥬 곡선을 오버레이하여 그립니다.
"""

import os
import numpy as np
import cv2
from pathlib import Path

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image


def load_image(image_path):
    """이미지 로드 (HEIF 지원) - 원본 크기 유지"""
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.heif', '.heic', '.hif']:
        with Image.open(image_path) as img:
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


def draw_lissajous_on_image(image, fx, fy, phase_deg, 
                             color=(0, 255, 255),  # Yellow (BGR)
                             thickness=3,
                             margin_ratio=0.15):
    """
    원본 이미지 위에 리사쥬 곡선을 그립니다.
    
    Args:
        image: 원본 이미지 (BGR)
        fx, fy: 주파수 쌍
        phase_deg: 위상차 (도)
        color: 선 색상 (BGR)
        thickness: 선 두께
        margin_ratio: 이미지 가장자리 여백 비율
    
    Returns:
        리사쥬 곡선이 그려진 이미지
    """
    from math import gcd
    
    h, w = image.shape[:2]
    result = image.copy()
    
    phase = np.radians(phase_deg)
    
    # 주기 계산
    g = gcd(int(fx * 1000), int(fy * 1000))
    lcm = (int(fx * 1000) * int(fy * 1000)) // g
    period = lcm / 1000.0 / fx
    
    # 곡선 생성
    t = np.linspace(0, period, 10000)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표로 변환
    # 중앙 기준, 여백 고려
    center_x, center_y = w // 2, h // 2
    scale = min(w, h) * (1 - margin_ratio) / 2
    
    x_img = (x * scale + center_x).astype(np.int32)
    y_img = (y * scale + center_y).astype(np.int32)
    
    # 곡선 그리기
    points = np.column_stack((x_img, y_img))
    
    for i in range(len(points) - 1):
        cv2.line(result, tuple(points[i]), tuple(points[i+1]), color, thickness)
    
    return result


def overlay_analysis(image_path, fx, fy, phase, output_path=None):
    """
    원본 이미지에 리사쥬 곡선을 오버레이합니다.
    """
    print(f"Loading: {image_path}")
    image = load_image(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Original size: {w}x{h}")
    print(f"Parameters: fx={fx}, fy={fy}, phase={phase}°")
    
    # 리사쥬 곡선 그리기 (노란색)
    result = draw_lissajous_on_image(
        image, fx, fy, phase,
        color=(0, 255, 255),  # Yellow
        thickness=max(3, w // 500)  # 이미지 크기에 비례한 두께
    )
    
    # 정보 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 1000
    text = f"fx={fx}, fy={fy}, phase={phase}deg"
    cv2.putText(result, text, (50, 100), font, font_scale, (0, 255, 255), 3)
    
    # 저장
    if output_path is None:
        base = Path(image_path).stem
        output_path = f"results/{base}_overlay.png"
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
    
    return result


if __name__ == '__main__':
    # 분석 결과에 따른 파라미터
    images_params = [
        ('results/IMG_0361.png', 3, 4, 90),   # 3:4 비율, 90도
        ('results/IMG_0383.png', 2, 1, 135),  # 2:1 비율, 135도
    ]
    
    for img_path, fx, fy, phase in images_params:
        if os.path.exists(img_path):
            overlay_analysis(img_path, fx, fy, phase)

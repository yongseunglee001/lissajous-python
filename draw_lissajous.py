#!/usr/bin/env python3
"""
1350Hz, 1480Hz, 위상차 0° 리사쥬 곡선 정확히 그리기
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import gcd


def draw_lissajous(fx, fy, phase_deg, size=800, save_path=None):
    """
    완전한 리사쥬 곡선 그리기
    """
    print(f"Drawing Lissajous: fx={fx}Hz, fy={fy}Hz, phase={phase_deg}°")
    
    # 주파수 비율 단순화
    g = gcd(int(fx), int(fy))
    ratio_x = fx // g
    ratio_y = fy // g
    print(f"Simplified ratio: {ratio_x}:{ratio_y}")
    print(f"GCD: {g}")
    
    # 완전한 패턴을 위한 시간 계산
    # LCM(fx, fy) / (fx * fy) = 1/GCD 초가 한 주기
    period = 1.0 / g  # 패턴이 완성되는 시간
    print(f"Pattern period: {period:.6f} seconds")
    
    # 샘플 생성 (충분히 많이)
    num_samples = 100000
    t = np.linspace(0, period, num_samples)
    
    phase = np.radians(phase_deg)
    x = np.sin(2 * np.pi * fx * t)
    y = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표 변환
    margin = 0.05
    scale = size * (1 - 2 * margin) / 2
    center = size // 2
    
    x_img = (x * scale + center).astype(np.int32)
    y_img = (y * scale + center).astype(np.int32)
    
    # 이미지 생성
    img = np.zeros((size, size), dtype=np.uint8)
    points = np.column_stack((x_img, y_img)).reshape(-1, 1, 2)
    cv2.polylines(img, [points], False, 255, 2, cv2.LINE_AA)
    
    # 컬러 이미지 (보기 좋게)
    img_color = np.zeros((size, size, 3), dtype=np.uint8)
    img_color[:, :, 1] = img  # Green
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # matplotlib으로 그린 곡선
    axes[0].plot(x, y, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('X (sin 2π·fx·t)')
    axes[0].set_ylabel('Y (sin 2π·fy·t + φ)')
    axes[0].set_title(f'Lissajous Curve\nfx={fx}Hz, fy={fy}Hz, phase={phase_deg}°\nRatio: {ratio_x}:{ratio_y}')
    
    # 이미지 버전
    axes[1].imshow(img_color)
    axes[1].set_title(f'Image Version ({size}x{size})\n{num_samples:,} samples')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # 이미지도 별도 저장
        img_path = save_path.replace('.png', '_image.png')
        cv2.imwrite(img_path, img_color)
        print(f"Saved: {img_path}")
    
    plt.show()
    
    return img


if __name__ == '__main__':
    # 1350Hz, 1480Hz, 위상차 0°
    fx = 1350
    fy = 1480
    phase = 0
    
    # 그리기
    draw_lissajous(fx, fy, phase, size=800, save_path='results/lissajous_1350_1480.png')

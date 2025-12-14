#!/usr/bin/env python3
"""
614 x 494 픽셀 그리드에 맞춘 리사주 궤적 시뮬레이션
x(t) = A * sin(2π * fx * t)
y(t) = B * sin(2π * fy * t + φ)

여기서 A와 B는 각각 그리드 크기에 맞게 조정됨
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_lissajous_on_grid(
    fx=1350, 
    fy=1480, 
    phase_deg=0, 
    duration=0.01,
    grid_width=614,
    grid_height=494,
    margin=10,
    output_path=None
):
    """
    지정된 그리드 크기에 맞춰 리사주 궤적을 시뮬레이션
    
    Args:
        fx, fy: 주파수 (Hz)
        phase_deg: 위상차 (도)
        duration: 시뮬레이션 시간 (초)
        grid_width: 그리드 X 크기 (픽셀) - 614
        grid_height: 그리드 Y 크기 (픽셀) - 494
        margin: 여백 (픽셀)
        output_path: 저장 경로 (None이면 저장 안 함)
    """
    phase = np.radians(phase_deg)
    
    # 중심점
    cx, cy = grid_width // 2, grid_height // 2
    
    # 진폭 조정: 그리드에 맞춤 (여백 제외)
    # X 진폭: (614 - 2*margin) / 2 = 297 픽셀
    # Y 진폭: (494 - 2*margin) / 2 = 237 픽셀
    amplitude_x = (grid_width - 2 * margin) / 2
    amplitude_y = (grid_height - 2 * margin) / 2
    
    # 샘플 수 (충분히 많이 - 고주파이므로)
    num_samples = int(max(fx, fy) * duration * 1000)
    num_samples = max(num_samples, 100000)  # 최소 100K 샘플
    
    t = np.linspace(0, duration, num_samples)
    
    # 리사주 방정식
    # x(t) = amplitude_x * sin(2π * fx * t)
    # y(t) = amplitude_y * sin(2π * fy * t + φ)
    x_norm = np.sin(2 * np.pi * fx * t)
    y_norm = np.sin(2 * np.pi * fy * t + phase)
    
    # 이미지 좌표로 변환
    x_img = x_norm * amplitude_x + cx
    y_img = y_norm * amplitude_y + cy
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1) 궤적 플롯 (연속 선)
    ax1 = axes[0]
    ax1.set_xlim(0, grid_width)
    ax1.set_ylim(grid_height, 0)  # Y축 반전 (이미지 좌표계)
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(x_img, y_img, 'lime', linewidth=0.5, alpha=0.8)
    ax1.scatter(cx, cy, c='red', s=100, marker='+', linewidths=2, zorder=5)
    ax1.set_title(f'Lissajous on {grid_width}×{grid_height} Grid\n'
                  f'fx={fx}Hz, fy={fy}Hz, φ={phase_deg}°, T={duration*1000:.1f}ms', 
                  fontsize=14)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3, color='gray')
    
    # 그리드 테두리
    rect = plt.Rectangle((0, 0), grid_width, grid_height, 
                          fill=False, edgecolor='white', linewidth=2)
    ax1.add_patch(rect)
    
    # 2) 정보 패널
    ax2 = axes[1]
    ax2.axis('off')
    
    info_text = f"""
    ┌─────────────────────────────────────────┐
    │     Lissajous Curve Simulation          │
    ├─────────────────────────────────────────┤
    │                                         │
    │  Grid Size:  {grid_width} × {grid_height} pixels        │
    │  Center:     ({cx}, {cy})                  │
    │                                         │
    │  X Amplitude: {amplitude_x:.1f} pixels              │
    │  Y Amplitude: {amplitude_y:.1f} pixels              │
    │                                         │
    │  Aspect Ratio: {grid_width/grid_height:.4f}                │
    │                                         │
    ├─────────────────────────────────────────┤
    │  Equation:                              │
    │                                         │
    │    x(t) = {amplitude_x:.0f} · sin(2π · {fx} · t)    │
    │    y(t) = {amplitude_y:.0f} · sin(2π · {fy} · t + {phase_deg}°) │
    │                                         │
    │  Frequency X: {fx} Hz                   │
    │  Frequency Y: {fy} Hz                   │
    │  Phase:       {phase_deg}°                        │
    │  Duration:    {duration*1000:.1f} ms                    │
    │                                         │
    │  Samples:     {num_samples:,}                   │
    └─────────────────────────────────────────┘
    """
    
    ax2.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top',
             family='monospace', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    
    return {
        'grid_size': (grid_width, grid_height),
        'center': (cx, cy),
        'amplitude_x': amplitude_x,
        'amplitude_y': amplitude_y,
        'num_samples': num_samples
    }


def simulate_on_pixel_canvas(
    fx=1350, 
    fy=1480, 
    phase_deg=0, 
    duration=0.01,
    grid_width=614,
    grid_height=494,
    margin=10,
    output_path=None
):
    """
    실제 픽셀 이미지로 리사주 궤적을 그림
    """
    import cv2
    
    phase = np.radians(phase_deg)
    
    cx, cy = grid_width // 2, grid_height // 2
    amplitude_x = (grid_width - 2 * margin) / 2
    amplitude_y = (grid_height - 2 * margin) / 2
    
    num_samples = int(max(fx, fy) * duration * 1000)
    num_samples = max(num_samples, 100000)
    
    t = np.linspace(0, duration, num_samples)
    
    x_norm = np.sin(2 * np.pi * fx * t)
    y_norm = np.sin(2 * np.pi * fy * t + phase)
    
    x_img = (x_norm * amplitude_x + cx).astype(np.int32)
    y_img = (y_norm * amplitude_y + cy).astype(np.int32)
    
    # 이미지 생성 (검정 배경)
    img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # 유효한 점들만
    valid = (x_img >= 0) & (x_img < grid_width) & (y_img >= 0) & (y_img < grid_height)
    x_valid = x_img[valid]
    y_valid = y_img[valid]
    
    # 점 그리기 (녹색 - 오실로스코프 스타일)
    points = np.column_stack((x_valid, y_valid)).reshape(-1, 1, 2)
    if len(points) > 1:
        cv2.polylines(img, [points], False, (0, 255, 0), 1, cv2.LINE_AA)
    
    # 중심 표시
    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Lissajous on {grid_width}×{grid_height} Pixel Canvas\n'
                 f'fx={fx}Hz, fy={fy}Hz, φ={phase_deg}°, T={duration*1000:.1f}ms',
                 fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        # 이미지 자체 저장
        cv2.imwrite(output_path, img)
        print(f"Saved pixel image: {output_path}")
        
        # 플롯도 저장
        fig_path = output_path.replace('.png', '_figure.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
    
    plt.close()
    
    # 통계
    unique_pixels = len(set(zip(x_valid.tolist(), y_valid.tolist())))
    total_pixels = grid_width * grid_height
    fill_rate = unique_pixels / total_pixels * 100
    
    print(f"\n[Statistics]")
    print(f"  Grid Size: {grid_width} × {grid_height} = {total_pixels:,} pixels")
    print(f"  Unique pixels covered: {unique_pixels:,}")
    print(f"  Fill Rate: {fill_rate:.2f}%")
    
    return img


if __name__ == '__main__':
    # 614 x 494 (사용자 지정 비율)
    print("=" * 60)
    print("Simulating Lissajous on 614 × 494 Grid")
    print("=" * 60)
    
    # 설정
    fx, fy = 1350, 1480  # 주파수
    phase = 0            # 위상차
    duration = 0.01      # 10ms
    
    # 1) 플롯으로 시뮬레이션
    result = simulate_lissajous_on_grid(
        fx=fx, fy=fy, phase_deg=phase, duration=duration,
        grid_width=614, grid_height=494, margin=10,
        output_path='results/lissajous_614x494_plot.png'
    )
    
    print(f"\nResult: {result}")
    
    # 2) 실제 픽셀 이미지로 그리기
    img = simulate_on_pixel_canvas(
        fx=fx, fy=fy, phase_deg=phase, duration=duration,
        grid_width=614, grid_height=494, margin=10,
        output_path='results/lissajous_614x494_pixels.png'
    )

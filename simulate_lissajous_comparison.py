#!/usr/bin/env python3
"""
614 x 494 픽셀 그리드에서 5개 스캐닝 시간별 리사주 궤적 비교
세로 배치, 업로드된 이미지 형태와 유사하게
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def generate_lissajous_image(fx, fy, phase_deg, duration, grid_width=614, grid_height=494, margin=10, 
                              line_color=(0, 0, 255), line_thickness=1):
    """리사주 이미지 생성"""
    phase = np.radians(phase_deg)
    
    cx, cy = grid_width // 2, grid_height // 2
    amplitude_x = (grid_width - 2 * margin) / 2
    amplitude_y = (grid_height - 2 * margin) / 2
    
    # 샘플 수 (duration에 비례)
    num_samples = int(max(fx, fy) * duration * 500)
    num_samples = max(num_samples, 10000)
    
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
    
    # 선 그리기
    points = np.column_stack((x_valid, y_valid)).reshape(-1, 1, 2)
    if len(points) > 1:
        cv2.polylines(img, [points], False, line_color, line_thickness, cv2.LINE_AA)
    
    return img


def create_comparison_figure(fx=1350, fy=1480, phase_deg=0, output_dir='results'):
    """5개 스캐닝 시간별 개별 이미지 저장"""
    
    durations = [1.0, 0.1, 0.04, 0.02, 0.01]
    
    grid_width = 614
    grid_height = 494
    
    saved_files = []
    
    for duration in durations:
        # 리사주 이미지 생성
        img = generate_lissajous_image(fx, fy, phase_deg, duration, grid_width, grid_height)
        
        # 파일명 생성 (1s, 0.5s, 0.1s, 0.04s, 0.01s)
        if duration >= 1.0:
            filename = f'{int(duration)}s.png'
        elif duration >= 0.1:
            filename = f'{duration:.1f}s.png'
        else:
            filename = f'{duration:.2f}s.png'
        
        filepath = f'{output_dir}/{filename}'
        cv2.imwrite(filepath, img)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_files


def create_side_by_side_comparison(fx=1350, fy=1480, phase_deg=0, output_path='results/lissajous_comparison_2col.png'):
    """2열 비교 (왼쪽: 이론, 오른쪽: 이론) - 업로드 이미지 스타일"""
    
    durations = [1.0, 0.1, 0.04, 0.02, 0.01]
    
    grid_width = 614
    grid_height = 494
    
    # Figure 생성 (5행 x 1열, 각 셀에 이미지 + 라벨)
    fig, axes = plt.subplots(5, 1, figsize=(10, 22))
    fig.patch.set_facecolor('black')
    
    for i, duration in enumerate(durations):
        ax = axes[i]
        
        img = generate_lissajous_image(fx, fy, phase_deg, duration, grid_width, grid_height)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_facecolor('black')
        ax.axis('off')
        
        if duration >= 0.1:
            title = f'Scanning time: {duration:.1f} [sec]'
        else:
            title = f'Scanning time: {duration:.2f} [sec]'
        ax.set_title(title, color='yellow', fontsize=16, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def create_short_duration_comparison(fx=1350, fy=1480, phase_deg=0, output_path='results/lissajous_short_durations.png'):
    """짧은 스캐닝 시간 비교 (0.004, 0.002, 0.001, 0.0005초)"""
    
    durations = [0.004, 0.002, 0.001, 0.0005]
    
    # 참조 이미지 크기에 맞춤 (334 x 251 픽셀)
    grid_width = 334
    grid_height = 251
    margin = 8
    
    # 밝은 빨간색 (BGR: 0, 50, 255 - 약간 오렌지 톤의 밝은 빨강)
    bright_red = (0, 50, 255)
    line_thickness = 1  # 1.5에 가깝게 (AA로 부드럽게)
    
    # 개별 이미지 저장 (캡션 없음)
    images = []
    for duration in durations:
        img = generate_lissajous_image(fx, fy, phase_deg, duration, grid_width, grid_height,
                                       margin, bright_red, line_thickness)
        images.append(img)
        
        if duration >= 0.001:
            filename = f'{duration:.3f}s.png'
        else:
            filename = f'{duration:.4f}s.png'
        filepath = f'results/{filename}'
        cv2.imwrite(filepath, img)
        print(f"Saved: {filepath}")
    
    # 4개 이미지를 세로로 이어붙이기 (캡션 없음)
    combined = np.vstack(images)
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")
    
    return output_path


def create_custom_comparison(fx, fy, phase_deg, durations, output_dir='results'):
    """주파수와 duration을 지정하여 이미지 생성 (파일명에 주파수, 위상 포함)"""
    
    grid_width = 334
    grid_height = 251
    margin = 8
    bright_red = (0, 50, 255)
    line_thickness = 1
    
    images = []
    for duration in durations:
        img = generate_lissajous_image(fx, fy, phase_deg, duration, grid_width, grid_height,
                                       margin, bright_red, line_thickness)
        images.append(img)
        
        # 파일명에 주파수, 위상 포함
        if duration >= 0.1:
            dur_str = f'{duration:.1f}s'
        elif duration >= 0.01:
            dur_str = f'{duration:.2f}s'
        elif duration >= 0.001:
            dur_str = f'{duration:.3f}s'
        else:
            dur_str = f'{duration:.4f}s'
        
        filename = f'{fx}_{fy}_p{int(phase_deg)}_{dur_str}.png'
        filepath = f'{output_dir}/{filename}'
        cv2.imwrite(filepath, img)
        print(f"Saved: {filepath}")
    
    # 합쳐진 이미지도 저장
    if len(images) > 1:
        combined = np.vstack(images)
        combined_path = f'{output_dir}/{fx}_{fy}_p{int(phase_deg)}_combined.png'
        cv2.imwrite(combined_path, combined)
        print(f"Saved: {combined_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Lissajous Scanning Time Comparison")
    print("Grid: 614 × 494 pixels")
    print("Frequencies: fx=1350Hz, fy=1480Hz")
    print("=" * 60)
    
    # 개별 이미지 생성 (1s, 0.1s, 0.04s, 0.02s, 0.01s)
    create_comparison_figure(
        fx=1350, fy=1480, phase_deg=0,
        output_dir='results'
    )
    
    # 짧은 스캐닝 시간 비교 (0.004s, 0.002s, 0.001s, 0.0005s)
    print("\n" + "=" * 60)
    print("Generating Short Duration Comparison")
    print("=" * 60)
    create_short_duration_comparison(
        fx=1350, fy=1480, phase_deg=0,
        output_path='results/lissajous_short_durations.png'
    )
    
    print("\nDone!")

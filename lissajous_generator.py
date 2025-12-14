"""
Lissajous Pattern Generator
주파수 쌍과 위상차를 기반으로 리사쥬 곡선을 생성합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LissajousGenerator:
    """리사쥬 곡선 생성기"""
    
    def __init__(self, freq_x: float, freq_y: float, phase_diff: float = 0.0,
                 amplitude_x: float = 1.0, amplitude_y: float = 1.0):
        """
        Args:
            freq_x: X축 주파수 (Hz)
            freq_y: Y축 주파수 (Hz)
            phase_diff: 위상차 (도, degrees)
            amplitude_x: X축 진폭
            amplitude_y: Y축 진폭
        """
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.phase_diff = np.radians(phase_diff)  # 라디안으로 변환
        self.amplitude_x = amplitude_x
        self.amplitude_y = amplitude_y
    
    def generate_curve(self, num_points: int = 10000, 
                       num_periods: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        리사쥬 곡선 좌표를 생성합니다.
        
        Args:
            num_points: 생성할 점의 수
            num_periods: 주기 수 (None이면 자동 계산)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (x좌표 배열, y좌표 배열)
        """
        # 완전한 패턴을 위한 시간 범위 계산
        if num_periods is None:
            # 두 주파수의 최소공배수 주기
            from math import gcd
            g = gcd(int(self.freq_x * 1000), int(self.freq_y * 1000))
            lcm = (int(self.freq_x * 1000) * int(self.freq_y * 1000)) // g
            period = lcm / 1000.0 / self.freq_x
        else:
            period = num_periods / min(self.freq_x, self.freq_y)
        
        t = np.linspace(0, period, num_points)
        
        # 리사쥬 방정식
        x = self.amplitude_x * np.sin(2 * np.pi * self.freq_x * t)
        y = self.amplitude_y * np.sin(2 * np.pi * self.freq_y * t + self.phase_diff)
        
        return x, y
    
    def generate_normalized_curve(self, num_points: int = 10000,
                                   image_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        0 ~ image_size 범위로 정규화된 좌표를 생성합니다.
        
        Args:
            num_points: 생성할 점의 수
            image_size: 이미지 크기 (픽셀)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 정규화된 (x좌표, y좌표)
        """
        x, y = self.generate_curve(num_points)
        
        # -1 ~ 1 범위를 0 ~ image_size로 변환 (여백 포함)
        margin = 0.1  # 10% 여백
        scale = (1 - margin) * image_size / 2
        
        x_norm = (x + 1) * scale + margin * image_size / 2
        y_norm = (y + 1) * scale + margin * image_size / 2
        
        return x_norm, y_norm
    
    def create_binary_image(self, size: int = 500, 
                            line_thickness: int = 2) -> np.ndarray:
        """
        리사쥬 곡선의 이진 이미지를 생성합니다.
        
        Args:
            size: 이미지 크기 (픽셀)
            line_thickness: 선 두께 (픽셀)
        
        Returns:
            np.ndarray: 이진 이미지 (0 또는 255)
        """
        import cv2
        
        x, y = self.generate_normalized_curve(num_points=10000, image_size=size)
        
        # 빈 이미지 생성
        image = np.zeros((size, size), dtype=np.uint8)
        
        # 좌표를 정수로 변환
        points = np.column_stack((x, y)).astype(np.int32)
        
        # 선 그리기
        for i in range(len(points) - 1):
            cv2.line(image, tuple(points[i]), tuple(points[i+1]), 255, line_thickness)
        
        return image
    
    def plot_curve(self, figsize: Tuple[int, int] = (8, 8), 
                   save_path: Optional[str] = None,
                   show: bool = True) -> None:
        """
        리사쥬 곡선을 플롯합니다.
        
        Args:
            figsize: 그림 크기
            save_path: 저장 경로 (None이면 저장하지 않음)
            show: 화면에 표시할지 여부
        """
        x, y = self.generate_curve()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, 'b-', linewidth=1.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Lissajous Curve\n'
                     f'fx={self.freq_x}Hz, fy={self.freq_y}Hz, '
                     f'φ={np.degrees(self.phase_diff):.1f}°')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_parameters_string(self) -> str:
        """파라미터 문자열 반환"""
        return (f"fx={self.freq_x}, fy={self.freq_y}, "
                f"phase={np.degrees(self.phase_diff):.1f}°")


def create_multiple_patterns(freq_pairs: list, phase_diffs: list,
                              save_dir: str = './patterns') -> None:
    """
    여러 주파수 쌍과 위상차에 대한 패턴을 생성합니다.
    
    Args:
        freq_pairs: [(fx1, fy1), (fx2, fy2), ...] 형태의 주파수 쌍 리스트
        phase_diffs: 위상차 리스트 (도)
        save_dir: 저장 디렉토리
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for fx, fy in freq_pairs:
        for phase in phase_diffs:
            gen = LissajousGenerator(fx, fy, phase)
            filename = f'lissajous_{fx}_{fy}_phase{phase}.png'
            save_path = os.path.join(save_dir, filename)
            gen.plot_curve(save_path=save_path, show=False)
            print(f'Generated: {filename}')


if __name__ == '__main__':
    # 예시: 3:2 주파수 비율, 45도 위상차
    gen = LissajousGenerator(freq_x=3, freq_y=2, phase_diff=45)
    gen.plot_curve()

"""
Curve Comparator for Lissajous Patterns
이론적 리사쥬 곡선과 추출된 곡선을 비교합니다.
"""

import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

from lissajous_generator import LissajousGenerator
from image_processor import LissajousImageProcessor


class CurveComparator:
    """리사쥬 곡선 비교기"""
    
    def __init__(self, theoretical_points: np.ndarray, 
                 extracted_points: np.ndarray):
        """
        Args:
            theoretical_points: 이론적 곡선 좌표 (N, 2)
            extracted_points: 추출된 곡선 좌표 (M, 2)
        """
        self.theoretical_points = theoretical_points
        self.extracted_points = extracted_points
    
    def hausdorff_distance(self) -> float:
        """
        Hausdorff 거리를 계산합니다.
        두 곡선 사이의 최대 거리를 측정합니다.
        
        Returns:
            float: Hausdorff 거리
        """
        if len(self.theoretical_points) == 0 or len(self.extracted_points) == 0:
            return float('inf')
        
        # 양방향 Hausdorff 거리
        h1 = directed_hausdorff(self.theoretical_points, self.extracted_points)[0]
        h2 = directed_hausdorff(self.extracted_points, self.theoretical_points)[0]
        
        return max(h1, h2)
    
    def mean_distance(self) -> Tuple[float, float]:
        """
        평균 거리를 계산합니다.
        
        Returns:
            Tuple[float, float]: (이론->추출 평균거리, 추출->이론 평균거리)
        """
        if len(self.theoretical_points) == 0 or len(self.extracted_points) == 0:
            return float('inf'), float('inf')
        
        # 이론 점들에서 가장 가까운 추출 점까지의 거리
        distances_to_extracted = []
        for pt in self.theoretical_points:
            dists = np.linalg.norm(self.extracted_points - pt, axis=1)
            distances_to_extracted.append(np.min(dists))
        
        # 추출 점들에서 가장 가까운 이론 점까지의 거리
        distances_to_theoretical = []
        for pt in self.extracted_points:
            dists = np.linalg.norm(self.theoretical_points - pt, axis=1)
            distances_to_theoretical.append(np.min(dists))
        
        return np.mean(distances_to_extracted), np.mean(distances_to_theoretical)
    
    def chamfer_distance(self) -> float:
        """
        Chamfer 거리를 계산합니다.
        두 점집합 간의 대칭적 평균 거리입니다.
        
        Returns:
            float: Chamfer 거리
        """
        mean_to_extracted, mean_to_theoretical = self.mean_distance()
        return (mean_to_extracted + mean_to_theoretical) / 2
    
    def coverage_score(self, threshold: float = 0.05) -> Tuple[float, float]:
        """
        커버리지 점수를 계산합니다.
        threshold 거리 내에 있는 점들의 비율입니다.
        
        Args:
            threshold: 거리 임계값 (정규화된 좌표 기준)
        
        Returns:
            Tuple[float, float]: (이론 커버리지, 추출 커버리지)
        """
        if len(self.theoretical_points) == 0 or len(self.extracted_points) == 0:
            return 0.0, 0.0
        
        # 이론 점들 중 추출 점에 가까운 비율
        theo_covered = 0
        for pt in self.theoretical_points:
            min_dist = np.min(np.linalg.norm(self.extracted_points - pt, axis=1))
            if min_dist <= threshold:
                theo_covered += 1
        
        # 추출 점들 중 이론 점에 가까운 비율
        ext_covered = 0
        for pt in self.extracted_points:
            min_dist = np.min(np.linalg.norm(self.theoretical_points - pt, axis=1))
            if min_dist <= threshold:
                ext_covered += 1
        
        theo_coverage = theo_covered / len(self.theoretical_points)
        ext_coverage = ext_covered / len(self.extracted_points)
        
        return theo_coverage, ext_coverage
    
    def f_score(self, threshold: float = 0.05) -> float:
        """
        F-Score를 계산합니다.
        정밀도와 재현율의 조화평균입니다.
        
        Args:
            threshold: 거리 임계값
        
        Returns:
            float: F-Score (0~1)
        """
        theo_coverage, ext_coverage = self.coverage_score(threshold)
        
        if theo_coverage + ext_coverage == 0:
            return 0.0
        
        # F1 Score
        return 2 * theo_coverage * ext_coverage / (theo_coverage + ext_coverage)
    
    def compute_all_metrics(self, threshold: float = 0.05) -> Dict[str, float]:
        """
        모든 비교 메트릭을 계산합니다.
        
        Args:
            threshold: 커버리지 계산용 임계값
        
        Returns:
            Dict: 모든 메트릭을 포함하는 딕셔너리
        """
        theo_coverage, ext_coverage = self.coverage_score(threshold)
        mean_to_ext, mean_to_theo = self.mean_distance()
        
        return {
            'hausdorff_distance': self.hausdorff_distance(),
            'chamfer_distance': self.chamfer_distance(),
            'mean_distance_to_extracted': mean_to_ext,
            'mean_distance_to_theoretical': mean_to_theo,
            'theoretical_coverage': theo_coverage,
            'extracted_coverage': ext_coverage,
            'f_score': self.f_score(threshold),
            'num_theoretical_points': len(self.theoretical_points),
            'num_extracted_points': len(self.extracted_points)
        }


class ImageBasedComparator:
    """이미지 기반 비교기 (픽셀 단위 비교)"""
    
    def __init__(self, theoretical_image: np.ndarray, 
                 extracted_image: np.ndarray):
        """
        Args:
            theoretical_image: 이론적 곡선 이미지 (이진)
            extracted_image: 추출된 곡선 이미지 (이진)
        """
        # 이미지 크기 맞추기
        if theoretical_image.shape != extracted_image.shape:
            h, w = max(theoretical_image.shape[0], extracted_image.shape[0]), \
                   max(theoretical_image.shape[1], extracted_image.shape[1])
            
            theo_resized = cv2.resize(theoretical_image, (w, h))
            ext_resized = cv2.resize(extracted_image, (w, h))
        else:
            theo_resized = theoretical_image
            ext_resized = extracted_image
        
        self.theoretical_image = (theo_resized > 127).astype(np.uint8)
        self.extracted_image = (ext_resized > 127).astype(np.uint8)
    
    def iou_score(self) -> float:
        """
        IoU (Intersection over Union) 점수를 계산합니다.
        
        Returns:
            float: IoU 점수 (0~1)
        """
        intersection = np.logical_and(self.theoretical_image, self.extracted_image)
        union = np.logical_or(self.theoretical_image, self.extracted_image)
        
        if np.sum(union) == 0:
            return 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def dice_coefficient(self) -> float:
        """
        Dice 계수를 계산합니다.
        
        Returns:
            float: Dice 계수 (0~1)
        """
        intersection = np.logical_and(self.theoretical_image, self.extracted_image)
        
        if np.sum(self.theoretical_image) + np.sum(self.extracted_image) == 0:
            return 0.0
        
        return 2 * np.sum(intersection) / \
               (np.sum(self.theoretical_image) + np.sum(self.extracted_image))
    
    def pixel_accuracy(self) -> float:
        """
        픽셀 정확도를 계산합니다.
        
        Returns:
            float: 픽셀 정확도 (0~1)
        """
        matches = (self.theoretical_image == self.extracted_image)
        return np.mean(matches)
    
    def distance_field_error(self) -> float:
        """
        거리장 기반 오차를 계산합니다.
        추출된 곡선 픽셀들의 이론적 곡선까지의 평균 거리입니다.
        
        Returns:
            float: 평균 거리 오차 (픽셀)
        """
        # 이론적 곡선의 거리장 계산
        distance_field = distance_transform_edt(1 - self.theoretical_image)
        
        # 추출된 곡선 픽셀들의 위치에서 거리 측정
        extracted_pixels = np.where(self.extracted_image > 0)
        
        if len(extracted_pixels[0]) == 0:
            return float('inf')
        
        distances = distance_field[extracted_pixels]
        return np.mean(distances)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        모든 이미지 기반 메트릭을 계산합니다.
        
        Returns:
            Dict: 모든 메트릭
        """
        return {
            'iou_score': self.iou_score(),
            'dice_coefficient': self.dice_coefficient(),
            'pixel_accuracy': self.pixel_accuracy(),
            'distance_field_error': self.distance_field_error(),
            'theoretical_pixel_count': np.sum(self.theoretical_image),
            'extracted_pixel_count': np.sum(self.extracted_image)
        }


def compare_lissajous_patterns(image_path: str, 
                                freq_x: float, freq_y: float, 
                                phase_diff: float,
                                visualize: bool = True,
                                save_path: Optional[str] = None) -> Dict:
    """
    이미지와 이론적 리사쥬 패턴을 비교합니다.
    
    Args:
        image_path: 비교할 이미지 경로
        freq_x: X축 주파수
        freq_y: Y축 주파수
        phase_diff: 위상차 (도)
        visualize: 시각화 여부
        save_path: 결과 저장 경로
    
    Returns:
        Dict: 비교 결과
    """
    # 1. 이론적 곡선 생성
    generator = LissajousGenerator(freq_x, freq_y, phase_diff)
    
    # 2. 이미지에서 곡선 추출
    processor = LissajousImageProcessor(image_path)
    processor.preprocess()
    processor.extract_skeleton()
    
    # 3. 이미지 크기에 맞춰 이론적 곡선 좌표 생성
    img_width, img_height = processor.get_image_size()
    img_size = max(img_width, img_height)
    
    theo_x, theo_y = generator.generate_normalized_curve(
        num_points=5000, image_size=img_size
    )
    theoretical_points = np.column_stack((theo_x, theo_y))
    
    # 4. 추출된 점들
    extracted_points = processor.get_curve_points()
    
    # 5. 정규화하여 비교
    # 정규화
    def normalize(points):
        if len(points) == 0:
            return points
        center = points.mean(axis=0)
        centered = points - center
        max_dist = np.max(np.abs(centered))
        return centered / max_dist if max_dist > 0 else centered
    
    theo_normalized = normalize(theoretical_points)
    ext_normalized = normalize(extracted_points)
    
    # 6. 점 기반 비교
    point_comparator = CurveComparator(theo_normalized, ext_normalized)
    point_metrics = point_comparator.compute_all_metrics()
    
    # 7. 이미지 기반 비교
    theo_image = generator.create_binary_image(size=img_size, line_thickness=2)
    
    if processor.skeleton is not None:
        ext_image = processor.skeleton
    else:
        ext_image = processor.processed_image
    
    image_comparator = ImageBasedComparator(theo_image, ext_image)
    image_metrics = image_comparator.compute_all_metrics()
    
    # 결과 통합
    results = {
        'parameters': {
            'freq_x': freq_x,
            'freq_y': freq_y,
            'phase_diff': phase_diff,
            'image_path': image_path
        },
        'point_based_metrics': point_metrics,
        'image_based_metrics': image_metrics
    }
    
    # 8. 시각화
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(processor.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 이론적 곡선
        axes[0, 1].imshow(theo_image, cmap='gray')
        axes[0, 1].set_title(f'Theoretical: fx={freq_x}, fy={freq_y}, φ={phase_diff}°')
        axes[0, 1].axis('off')
        
        # 추출된 곡선
        axes[0, 2].imshow(ext_image, cmap='gray')
        axes[0, 2].set_title('Extracted Curve')
        axes[0, 2].axis('off')
        
        # 오버레이 비교
        overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # 리사이즈
        theo_resized = cv2.resize(theo_image, (img_size, img_size))
        ext_resized = cv2.resize(ext_image, (img_size, img_size))
        overlay[:, :, 0] = theo_resized  # 빨강 = 이론
        overlay[:, :, 1] = ext_resized   # 초록 = 추출
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (Red=Theory, Green=Extracted)')
        axes[1, 0].axis('off')
        
        # 정규화된 점들 비교
        axes[1, 1].scatter(theo_normalized[:, 0], theo_normalized[:, 1], 
                          s=1, c='red', alpha=0.3, label='Theoretical')
        if len(ext_normalized) > 0:
            axes[1, 1].scatter(ext_normalized[:, 0], ext_normalized[:, 1], 
                              s=1, c='blue', alpha=0.3, label='Extracted')
        axes[1, 1].legend()
        axes[1, 1].set_title('Normalized Point Comparison')
        axes[1, 1].set_aspect('equal')
        
        # 메트릭 표시
        metrics_text = (
            f"Point Metrics:\n"
            f"  Hausdorff Distance: {point_metrics['hausdorff_distance']:.4f}\n"
            f"  Chamfer Distance: {point_metrics['chamfer_distance']:.4f}\n"
            f"  F-Score: {point_metrics['f_score']:.4f}\n\n"
            f"Image Metrics:\n"
            f"  IoU Score: {image_metrics['iou_score']:.4f}\n"
            f"  Dice Coefficient: {image_metrics['dice_coefficient']:.4f}\n"
            f"  Distance Field Error: {image_metrics['distance_field_error']:.2f}px"
        )
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center', family='monospace')
        axes[1, 2].set_title('Comparison Metrics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 5:
        image_path = sys.argv[1]
        freq_x = float(sys.argv[2])
        freq_y = float(sys.argv[3])
        phase_diff = float(sys.argv[4])
        
        results = compare_lissajous_patterns(
            image_path, freq_x, freq_y, phase_diff
        )
        
        print("\n=== Comparison Results ===")
        print(f"Parameters: fx={freq_x}, fy={freq_y}, phase={phase_diff}°")
        print("\nPoint-based Metrics:")
        for key, value in results['point_based_metrics'].items():
            print(f"  {key}: {value:.4f}")
        print("\nImage-based Metrics:")
        for key, value in results['image_based_metrics'].items():
            print(f"  {key}: {value:.4f}")
    else:
        print("Usage: python curve_comparator.py <image_path> <freq_x> <freq_y> <phase_diff>")
        print("Example: python curve_comparator.py pattern.png 3 2 45")

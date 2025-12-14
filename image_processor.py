"""
Image Processor for Lissajous Pattern Extraction
업로드된 이미지에서 리사쥬 궤적을 추출합니다.
"""

import numpy as np
import cv2
from skimage import morphology, feature
from scipy import ndimage
from typing import Tuple, Optional, List
from PIL import Image


class LissajousImageProcessor:
    """리사쥬 패턴 이미지 처리기"""
    
    def __init__(self, image_path: str):
        """
        Args:
            image_path: 이미지 파일 경로
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = None
        self.skeleton = None
        self.contours = None
    
    def preprocess(self, blur_kernel: int = 5, 
                   morph_kernel: int = 3) -> np.ndarray:
        """
        이미지 전처리 (노이즈 제거, 이진화)
        
        Args:
            blur_kernel: 가우시안 블러 커널 크기
            morph_kernel: 모폴로지 연산 커널 크기
        
        Returns:
            np.ndarray: 전처리된 이진 이미지
        """
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(self.gray_image, (blur_kernel, blur_kernel), 0)
        
        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        self.processed_image = binary
        return binary
    
    def extract_skeleton(self) -> np.ndarray:
        """
        이진 이미지에서 골격(skeleton)을 추출합니다.
        곡선의 중심선을 얻기 위함입니다.
        
        Returns:
            np.ndarray: 골격화된 이진 이미지
        """
        if self.processed_image is None:
            self.preprocess()
        
        # 이진 이미지를 bool로 변환
        binary_bool = self.processed_image > 0
        
        # 골격화
        self.skeleton = morphology.skeletonize(binary_bool).astype(np.uint8) * 255
        
        return self.skeleton
    
    def extract_edge_curve(self, method: str = 'canny') -> np.ndarray:
        """
        에지 검출을 통해 곡선을 추출합니다.
        
        Args:
            method: 'canny' 또는 'sobel'
        
        Returns:
            np.ndarray: 에지 이미지
        """
        if method == 'canny':
            edges = cv2.Canny(self.gray_image, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return edges
    
    def extract_contours(self) -> List[np.ndarray]:
        """
        이미지에서 컨투어를 추출합니다.
        
        Returns:
            List[np.ndarray]: 컨투어 리스트
        """
        if self.processed_image is None:
            self.preprocess()
        
        contours, _ = cv2.findContours(
            self.processed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # 가장 긴 컨투어를 선택 (보통 리사쥬 곡선)
        self.contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return self.contours
    
    def get_curve_points(self, use_skeleton: bool = True) -> np.ndarray:
        """
        곡선의 좌표 점들을 추출합니다.
        
        Args:
            use_skeleton: 골격화 사용 여부
        
        Returns:
            np.ndarray: (N, 2) 형태의 좌표 배열
        """
        if use_skeleton:
            if self.skeleton is None:
                self.extract_skeleton()
            # 골격에서 좌표 추출
            points = np.column_stack(np.where(self.skeleton > 0))
            # (row, col) -> (x, y) 변환
            points = points[:, ::-1]
        else:
            if self.contours is None:
                self.extract_contours()
            if len(self.contours) > 0:
                # 가장 큰 컨투어의 점들
                points = self.contours[0].reshape(-1, 2)
            else:
                points = np.array([])
        
        return points
    
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        좌표를 -1 ~ 1 범위로 정규화합니다.
        
        Args:
            points: (N, 2) 형태의 좌표 배열
        
        Returns:
            np.ndarray: 정규화된 좌표 배열
        """
        if len(points) == 0:
            return points
        
        # 중심점 계산
        center = points.mean(axis=0)
        
        # 중심으로 이동
        centered = points - center
        
        # 스케일링 (최대 거리로 정규화)
        max_dist = np.max(np.abs(centered))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized
    
    def visualize_extraction(self, save_path: Optional[str] = None) -> None:
        """
        추출 과정을 시각화합니다.
        
        Args:
            save_path: 저장 경로
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 그레이스케일
        axes[0, 1].imshow(self.gray_image, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # 전처리 결과
        if self.processed_image is None:
            self.preprocess()
        axes[0, 2].imshow(self.processed_image, cmap='gray')
        axes[0, 2].set_title('Binary (After Preprocessing)')
        axes[0, 2].axis('off')
        
        # 골격
        if self.skeleton is None:
            self.extract_skeleton()
        axes[1, 0].imshow(self.skeleton, cmap='gray')
        axes[1, 0].set_title('Skeleton')
        axes[1, 0].axis('off')
        
        # 에지 검출 (Canny)
        edges = self.extract_edge_curve('canny')
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Canny Edge Detection')
        axes[1, 1].axis('off')
        
        # 추출된 점들
        points = self.get_curve_points()
        axes[1, 2].scatter(points[:, 0], points[:, 1], s=1, c='blue', alpha=0.5)
        axes[1, 2].set_title(f'Extracted Points ({len(points)} points)')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_image_size(self) -> Tuple[int, int]:
        """이미지 크기 반환 (width, height)"""
        return self.original_image.shape[1], self.original_image.shape[0]


class ColorBasedExtractor:
    """
    색상 기반 곡선 추출기
    특정 색상의 선을 추출할 때 사용합니다.
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
    
    def extract_by_color(self, lower_hsv: Tuple[int, int, int],
                          upper_hsv: Tuple[int, int, int]) -> np.ndarray:
        """
        특정 HSV 색상 범위의 영역을 추출합니다.
        
        Args:
            lower_hsv: HSV 하한값 (H, S, V)
            upper_hsv: HSV 상한값 (H, S, V)
        
        Returns:
            np.ndarray: 이진 마스크
        """
        lower = np.array(lower_hsv)
        upper = np.array(upper_hsv)
        mask = cv2.inRange(self.hsv, lower, upper)
        return mask
    
    def extract_green_curve(self) -> np.ndarray:
        """녹색 곡선 추출 (오실로스코프 디스플레이용)"""
        return self.extract_by_color((35, 50, 50), (85, 255, 255))
    
    def extract_white_curve(self) -> np.ndarray:
        """흰색/밝은 곡선 추출"""
        return self.extract_by_color((0, 0, 200), (180, 30, 255))
    
    def extract_bright_regions(self, threshold: int = 200) -> np.ndarray:
        """밝은 영역 추출 (그레이스케일 기반)"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        processor = LissajousImageProcessor(image_path)
        processor.visualize_extraction()
    else:
        print("Usage: python image_processor.py <image_path>")
        print("이미지 경로를 지정해주세요.")

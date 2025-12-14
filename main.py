#!/usr/bin/env python3
"""
Lissajous Pattern Analysis - Main Script
리사쥬 패턴 생성, 추출, 비교를 위한 메인 실행 스크립트입니다.

사용법:
    python main.py --image <이미지경로> --fx <주파수X> --fy <주파수Y> --phase <위상차>
    python main.py --interactive
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# HEIF 이미지 지원
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False
    print("Warning: pillow-heif not installed. HEIF/HEIC images not supported.")
    print("Install with: pip install pillow-heif")

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정 (Mac)
try:
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'AppleGothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

from lissajous_generator import LissajousGenerator
from image_processor import LissajousImageProcessor, ColorBasedExtractor
from curve_comparator import compare_lissajous_patterns, CurveComparator, ImageBasedComparator


def convert_heif_to_png(heif_path: str, output_dir: str = None) -> str:
    """
    HEIF/HEIC 이미지를 PNG로 변환합니다.
    
    Args:
        heif_path: HEIF 이미지 경로
        output_dir: 출력 디렉토리 (None이면 같은 디렉토리)
    
    Returns:
        str: 변환된 PNG 파일 경로
    """
    if not HEIF_SUPPORTED:
        raise ImportError("pillow-heif가 설치되어 있지 않습니다. pip install pillow-heif")
    
    heif_path = Path(heif_path)
    
    if output_dir is None:
        output_dir = heif_path.parent
    
    output_path = Path(output_dir) / f"{heif_path.stem}.png"
    
    # HEIF 열고 PNG로 저장
    with Image.open(heif_path) as img:
        img.save(output_path, 'PNG')
    
    print(f"변환 완료: {heif_path} -> {output_path}")
    return str(output_path)


def load_image(image_path: str) -> np.ndarray:
    """
    다양한 포맷의 이미지를 로드합니다. HEIF/HEIC 포맷도 지원합니다.
    
    Args:
        image_path: 이미지 경로
    
    Returns:
        np.ndarray: BGR 이미지 배열
    """
    path = Path(image_path)
    extension = path.suffix.lower()
    
    if extension in ['.heif', '.heic', '.hif']:
        if not HEIF_SUPPORTED:
            raise ImportError("HEIF 이미지를 열기 위해 pillow-heif를 설치해주세요: pip install pillow-heif")
        
        # PIL로 HEIF 열기
        with Image.open(image_path) as img:
            # RGB -> BGR 변환 (OpenCV 형식)
            rgb_array = np.array(img)
            if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            else:
                bgr_array = rgb_array
            return bgr_array
    else:
        # 일반 이미지는 OpenCV로 로드
        return cv2.imread(image_path)


def analyze_single_image(image_path: str, freq_x: float, freq_y: float, 
                         phase_diff: float, output_dir: str = './results'):
    """
    단일 이미지를 분석합니다.
    
    Args:
        image_path: 이미지 경로
        freq_x: X축 주파수
        freq_y: Y축 주파수  
        phase_diff: 위상차 (도)
        output_dir: 결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"리사쥬 패턴 분석")
    print(f"{'='*60}")
    print(f"이미지: {image_path}")
    print(f"파라미터: fx={freq_x}, fy={freq_y}, phase={phase_diff}°")
    
    # 이미지 로드
    image = load_image(image_path)
    if image is None:
        print(f"Error: 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 임시 PNG로 저장 (이미지 프로세서가 경로로 작동하므로)
    temp_path = os.path.join(output_dir, 'temp_input.png')
    cv2.imwrite(temp_path, image)
    
    # 이론적 곡선 생성
    generator = LissajousGenerator(freq_x, freq_y, phase_diff)
    
    # 이미지 크기
    img_height, img_width = image.shape[:2]
    img_size = max(img_width, img_height)
    
    # 이론적 곡선 이미지 생성
    theo_image = generator.create_binary_image(size=img_size, line_thickness=3)
    
    # 이미지 처리 및 곡선 추출
    processor = LissajousImageProcessor(temp_path)
    
    # 전처리 시각화
    print("\n추출 과정 시각화...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 원본
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('원본 이미지')
    axes[0, 0].axis('off')
    
    # 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('그레이스케일')
    axes[0, 1].axis('off')
    
    # 전처리 (이진화)
    binary = processor.preprocess()
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('이진화')
    axes[0, 2].axis('off')
    
    # 골격화
    skeleton = processor.extract_skeleton()
    axes[0, 3].imshow(skeleton, cmap='gray')
    axes[0, 3].set_title('골격화 (Skeleton)')
    axes[0, 3].axis('off')
    
    # 이론 곡선
    axes[1, 0].imshow(theo_image, cmap='gray')
    axes[1, 0].set_title(f'이론 곡선\n(fx={freq_x}, fy={freq_y}, φ={phase_diff}°)')
    axes[1, 0].axis('off')
    
    # 추출된 점들
    extracted_points = processor.get_curve_points()
    if len(extracted_points) > 0:
        axes[1, 1].scatter(extracted_points[:, 0], extracted_points[:, 1], 
                          s=0.5, c='blue', alpha=0.5)
        axes[1, 1].set_title(f'추출된 점들 ({len(extracted_points):,}개)')
    else:
        axes[1, 1].text(0.5, 0.5, '점 추출 실패', ha='center', va='center')
        axes[1, 1].set_title('추출된 점들')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    # 오버레이 비교
    overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # 리사이즈
    if skeleton.shape[0] != img_size or skeleton.shape[1] != img_size:
        skeleton_resized = cv2.resize(skeleton, (img_size, img_size))
    else:
        skeleton_resized = skeleton
    overlay[:, :, 2] = theo_image  # Red = 이론
    overlay[:, :, 1] = skeleton_resized  # Green = 추출
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('오버레이 비교\n(빨강=이론, 초록=추출)')
    axes[1, 2].axis('off')
    
    # 메트릭 계산 및 표시
    image_comparator = ImageBasedComparator(theo_image, skeleton_resized)
    metrics = image_comparator.compute_all_metrics()
    
    metrics_text = (
        f"이미지 기반 메트릭:\n\n"
        f"  IoU Score: {metrics['iou_score']:.4f}\n"
        f"  Dice Coefficient: {metrics['dice_coefficient']:.4f}\n"
        f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n"
        f"  Distance Error: {metrics['distance_field_error']:.2f}px\n\n"
        f"  이론 픽셀 수: {metrics['theoretical_pixel_count']:,}\n"
        f"  추출 픽셀 수: {metrics['extracted_pixel_count']:,}"
    )
    axes[1, 3].text(0.1, 0.5, metrics_text, fontsize=11, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 3].set_title('비교 메트릭')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    # 결과 저장
    base_name = Path(image_path).stem
    result_path = os.path.join(output_dir, f'{base_name}_analysis.png')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"분석 결과 저장: {result_path}")
    
    plt.show()
    
    # 정리
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return metrics


def find_best_parameters(image_path: str, 
                         freq_pairs: list = None,
                         phase_range: tuple = (0, 180, 15),
                         output_dir: str = './results'):
    """
    이미지에 가장 잘 맞는 주파수 쌍과 위상차를 찾습니다.
    
    Args:
        image_path: 이미지 경로
        freq_pairs: [(fx, fy), ...] 형태의 주파수 쌍 리스트
        phase_range: (시작, 끝, 간격) 위상차 범위
        output_dir: 결과 저장 디렉토리
    
    Returns:
        dict: 최적 파라미터와 점수
    """
    if freq_pairs is None:
        # 기본 주파수 쌍들
        freq_pairs = [
            (1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
            (2, 3), (3, 2), (3, 4), (4, 3), (3, 5), (5, 3)
        ]
    
    phases = np.arange(phase_range[0], phase_range[1] + 1, phase_range[2])
    
    print(f"\n{'='*60}")
    print(f"최적 파라미터 탐색")
    print(f"{'='*60}")
    print(f"주파수 쌍: {len(freq_pairs)}개")
    print(f"위상차 범위: {phase_range[0]}° ~ {phase_range[1]}° (간격 {phase_range[2]}°)")
    print(f"총 조합: {len(freq_pairs) * len(phases)}개")
    
    # 이미지 로드
    image = load_image(image_path)
    if image is None:
        print(f"Error: 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    temp_path = os.path.join(output_dir, 'temp_input.png')
    cv2.imwrite(temp_path, image)
    
    # 이미지 처리
    processor = LissajousImageProcessor(temp_path)
    processor.preprocess()
    skeleton = processor.extract_skeleton()
    img_size = max(image.shape[:2])
    
    if skeleton.shape[0] != img_size or skeleton.shape[1] != img_size:
        skeleton_resized = cv2.resize(skeleton, (img_size, img_size))
    else:
        skeleton_resized = skeleton
    
    best_score = -1
    best_params = None
    all_results = []
    
    for fx, fy in freq_pairs:
        for phase in phases:
            generator = LissajousGenerator(fx, fy, phase)
            theo_image = generator.create_binary_image(size=img_size, line_thickness=3)
            
            comparator = ImageBasedComparator(theo_image, skeleton_resized)
            metrics = comparator.compute_all_metrics()
            
            # IoU와 Dice의 평균을 점수로 사용
            score = (metrics['iou_score'] + metrics['dice_coefficient']) / 2
            
            all_results.append({
                'freq_x': fx,
                'freq_y': fy,
                'phase': phase,
                'score': score,
                'iou': metrics['iou_score'],
                'dice': metrics['dice_coefficient']
            })
            
            if score > best_score:
                best_score = score
                best_params = (fx, fy, phase)
    
    print(f"\n최적 파라미터:")
    print(f"  주파수: fx={best_params[0]}, fy={best_params[1]}")
    print(f"  위상차: {best_params[2]}°")
    print(f"  점수: {best_score:.4f}")
    
    # 정리
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return {
        'best_params': {
            'freq_x': best_params[0],
            'freq_y': best_params[1],
            'phase': best_params[2]
        },
        'best_score': best_score,
        'all_results': all_results
    }


def interactive_mode():
    """대화형 모드로 분석을 진행합니다."""
    print("\n" + "="*60)
    print("리사쥬 패턴 분석 - 대화형 모드")
    print("="*60)
    
    # 현재 디렉토리의 이미지 파일 찾기
    current_dir = Path('.')
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.heif', '*.heic', '*.hif', '*.HIF']:
        image_files.extend(current_dir.glob(ext))
    
    if not image_files:
        print("현재 디렉토리에 이미지 파일이 없습니다.")
        image_path = input("이미지 경로를 입력하세요: ").strip()
    else:
        print("\n발견된 이미지 파일:")
        for i, f in enumerate(image_files):
            print(f"  [{i}] {f.name}")
        
        choice = input("\n분석할 이미지 번호 또는 경로를 입력하세요: ").strip()
        
        try:
            idx = int(choice)
            image_path = str(image_files[idx])
        except ValueError:
            image_path = choice
    
    # 분석 모드 선택
    print("\n분석 모드:")
    print("  [1] 직접 파라미터 입력")
    print("  [2] 최적 파라미터 자동 탐색")
    
    mode = input("\n모드를 선택하세요 (1 또는 2): ").strip()
    
    if mode == '1':
        freq_x = float(input("X축 주파수 (fx): "))
        freq_y = float(input("Y축 주파수 (fy): "))
        phase_diff = float(input("위상차 (도): "))
        
        analyze_single_image(image_path, freq_x, freq_y, phase_diff)
    else:
        result = find_best_parameters(image_path)
        if result:
            print("\n최적 파라미터로 상세 분석을 진행합니다...")
            params = result['best_params']
            analyze_single_image(image_path, params['freq_x'], params['freq_y'], params['phase'])


def main():
    parser = argparse.ArgumentParser(
        description='리사쥬 패턴 분석 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 특정 파라미터로 분석
    python main.py --image pattern.png --fx 3 --fy 2 --phase 45
    
    # 최적 파라미터 자동 탐색
    python main.py --image pattern.png --auto
    
    # 대화형 모드
    python main.py --interactive
    
    # HEIF 이미지 변환만
    python main.py --convert image.heif
"""
    )
    
    parser.add_argument('--image', '-i', type=str, help='분석할 이미지 경로')
    parser.add_argument('--fx', type=float, help='X축 주파수')
    parser.add_argument('--fy', type=float, help='Y축 주파수')
    parser.add_argument('--phase', type=float, default=0, help='위상차 (도, 기본값: 0)')
    parser.add_argument('--auto', action='store_true', help='최적 파라미터 자동 탐색')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    parser.add_argument('--convert', type=str, help='HEIF 이미지를 PNG로 변환')
    parser.add_argument('--output', '-o', type=str, default='./results', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # HEIF 변환 모드
    if args.convert:
        convert_heif_to_png(args.convert, args.output)
        return
    
    # 대화형 모드
    if args.interactive:
        interactive_mode()
        return
    
    # 이미지 필수 체크
    if not args.image:
        parser.print_help()
        print("\n오류: --image 또는 --interactive 옵션을 지정해주세요.")
        return
    
    # 자동 탐색 모드
    if args.auto:
        result = find_best_parameters(args.image, output_dir=args.output)
        if result:
            params = result['best_params']
            analyze_single_image(args.image, params['freq_x'], params['freq_y'], 
                               params['phase'], args.output)
        return
    
    # 직접 파라미터 지정 모드
    if args.fx is None or args.fy is None:
        parser.print_help()
        print("\n오류: --fx와 --fy를 지정하거나 --auto 옵션을 사용해주세요.")
        return
    
    analyze_single_image(args.image, args.fx, args.fy, args.phase, args.output)


if __name__ == '__main__':
    main()

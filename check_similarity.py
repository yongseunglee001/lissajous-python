
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def check_similarity():
    # 파일 경로
    img_01 = cv2.imread('results/0.1s.png')
    img_05 = cv2.imread('results/0.5s.png')
    img_10 = cv2.imread('results/1s.png')
    
    if img_01 is None or img_05 is None or img_10 is None:
        print("Error: Images not found. Please run the simulation first.")
        return

    # 차이 계산
    diff_01_05 = cv2.absdiff(img_01, img_05)
    diff_01_10 = cv2.absdiff(img_01, img_10)
    
    # 텍스트로 결과 출력
    full_match_05 = np.all(diff_01_05 == 0)
    full_match_10 = np.all(diff_01_10 == 0)
    
    print(f"0.1s vs 0.5s Identical: {full_match_05}")
    if not full_match_05:
        print(f"  Max diff: {np.max(diff_01_05)}")
        print(f"  Mean diff: {np.mean(diff_01_05)}")
        
    print(f"0.1s vs 1.0s Identical: {full_match_10}")
    if not full_match_10:
        print(f"  Max diff: {np.max(diff_01_10)}")
        print(f"  Mean diff: {np.mean(diff_01_10)}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 0.1s (Base)
    axes[0].imshow(cv2.cvtColor(img_01, cv2.COLOR_BGR2RGB))
    axes[0].set_title('0.1s (Base)')
    axes[0].axis('off')
    
    # Diff 0.5s
    axes[1].imshow(diff_01_05 * 10) # 잘 보이게 증폭
    axes[1].set_title('Diff with 0.5s (x10 amplified)')
    axes[1].axis('off')
    
    # Diff 1.0s
    axes[2].imshow(diff_01_10 * 10) # 잘 보이게 증폭
    axes[2].set_title('Diff with 1.0s (x10 amplified)')
    axes[2].axis('off')
    
    result_path = 'results/consistency_check.png'
    plt.savefig(result_path)
    print(f"Saved comparison to {result_path}")

if __name__ == "__main__":
    check_similarity()

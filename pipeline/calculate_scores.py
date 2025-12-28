"""
計算推薦方法分數
=================================================

功能：
讀取填好評分的 CSV 檔案，計算每種方法的加權總分。

計分方式：
- 排名 1 的權重：5
- 排名 2 的權重：4
- 排名 3 的權重：3
- 排名 4 的權重：2
- 排名 5 的權重：1

總分 = Σ(該排名所有使用者的平均評分 × 權重)

使用方式：
  python pipeline/calculate_scores.py evaluation_ratings.csv
  python pipeline/calculate_scores.py evaluation_ratings.csv --output scores.json
"""

import csv
import json
import argparse
import numpy as np
from typing import Dict, List
from collections import defaultdict

# 排名權重
RANK_WEIGHTS = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}

# 方法名稱映射
METHOD_HEADERS = {
    "BM25_only_rank": "BM25 only",
    "BM25_Emotion_rank": "BM25 + Emotion reranking",
    "BM25_Topic_rank": "BM25 + Topic reranking",
    "BM25_Emotion_Topic_rank": "BM25 + Emotion + Topic reranking",
    "BM25_Emotion_Topic_PPR_rank": "BM25 + Emotion + Topic + PPR",
    "Emotion_Topic_only_rank": "Emotion + Topic only (no BM25)",
    "BM25_PPR_rank": "BM25 + PPR (no reranking)",
    "PPR_only_rank": "PPR only (similarity-based)",
}

def parse_rating(value: str) -> float:
    """解析評分（支援空值、數字、小數）"""
    if not value or value.strip() == "":
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None

def calculate_method_scores(csv_path: str) -> Dict:
    """計算每種方法的分數"""
    # 讀取 CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 找出所有評分欄位
    rating_columns = [col for col in reader.fieldnames if col.startswith("rating_user")]
    print(f"[Info] Found {len(rating_columns)} rating columns: {rating_columns}")
    
    # 統計每種方法在各排名的評分
    method_stats = defaultdict(lambda: defaultdict(list))  # method -> rank -> [ratings]
    
    for row in rows:
        # 讀取各方法的排名
        for header, method_name in METHOD_HEADERS.items():
            rank_str = row.get(header, "").strip()
            if rank_str and rank_str.isdigit():
                rank = int(rank_str)
                if 1 <= rank <= 5:  # 只考慮 top-5
                    # 讀取所有使用者的評分
                    ratings = []
                    for rating_col in rating_columns:
                        rating = parse_rating(row.get(rating_col, ""))
                        if rating is not None:
                            ratings.append(rating)
                    
                    # 如果有評分，記錄到對應的排名
                    if ratings:
                        method_stats[method_name][rank].extend(ratings)
    
    # 計算每種方法的總分
    method_scores = {}
    method_details = {}
    
    for method_name, rank_ratings in method_stats.items():
        total_score = 0.0
        details = {}
        
        for rank in range(1, 6):  # 排名 1-5
            ratings = rank_ratings.get(rank, [])
            if ratings:
                avg_rating = np.mean(ratings)
                weight = RANK_WEIGHTS[rank]
                weighted_score = avg_rating * weight
                total_score += weighted_score
                
                details[rank] = {
                    "ratings": ratings,
                    "avg_rating": float(avg_rating),
                    "weight": weight,
                    "weighted_score": float(weighted_score),
                    "count": len(ratings)
                }
            else:
                details[rank] = {
                    "ratings": [],
                    "avg_rating": None,
                    "weight": RANK_WEIGHTS[rank],
                    "weighted_score": 0.0,
                    "count": 0
                }
        
        method_scores[method_name] = float(total_score)
        method_details[method_name] = details
    
    return {
        "method_scores": method_scores,
        "method_details": method_details,
        "num_users": len(rating_columns),
        "num_songs": len(rows),
    }

def print_results(results: Dict):
    """輸出結果"""
    print("\n" + "=" * 80)
    print("推薦方法分數計算結果")
    print("=" * 80)
    
    print(f"\n統計資訊：")
    print(f"  - 評分使用者數：{results['num_users']}")
    print(f"  - 歌曲總數：{results['num_songs']}")
    
    print(f"\n各方法總分（按分數排序）：")
    print("-" * 80)
    
    # 按分數排序
    sorted_methods = sorted(
        results['method_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for rank, (method_name, total_score) in enumerate(sorted_methods, 1):
        print(f"\n{rank}. {method_name}")
        print(f"   總分：{total_score:.2f}")
        
        # 顯示各排名的詳細資訊
        details = results['method_details'][method_name]
        print(f"   各排名詳細：")
        for r in range(1, 6):
            d = details[r]
            if d['count'] > 0:
                print(f"     排名 {r} (權重 {d['weight']}): "
                      f"平均評分 {d['avg_rating']:.2f} × {d['weight']} = {d['weighted_score']:.2f} "
                      f"({d['count']} 個評分)")
            else:
                print(f"     排名 {r} (權重 {d['weight']}): 無評分")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Calculate recommendation method scores")
    parser.add_argument("csv_path", type=str, help="Path to CSV file with ratings")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 計算分數
    results = calculate_method_scores(args.csv_path)
    
    # 輸出結果
    print_results(results)
    
    # 保存結果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Scores saved to {args.output}")

if __name__ == "__main__":
    main()



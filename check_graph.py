"""快速檢查圖譜是否正常"""
import numpy as np
from scipy.sparse import load_npz
import json

# 載入圖譜
print("=" * 60)
print("載入圖譜...")
graph = load_npz('outputs/graph/song_graph.npz')
print(f"✅ 圖譜載入成功")

# 基本資訊
print("\n" + "=" * 60)
print("圖譜基本資訊")
print("=" * 60)
print(f"Shape: {graph.shape}")
print(f"邊數 (nnz): {graph.nnz:,}")
print(f"資料型態: {graph.dtype}")
print(f"格式: {graph.format}")

# 檢查前 5 首歌的鄰居
print("\n" + "=" * 60)
print("檢查前 5 首歌的鄰居（驗證稀疏化是否正常）")
print("=" * 60)
for i in range(min(5, graph.shape[0])):
    row = graph.getrow(i)
    if row.nnz > 0:
        # 取得鄰居和相似度
        neighbors = row.indices
        similarities = row.data
        # 排序（由高到低）
        sorted_idx = np.argsort(similarities)[::-1]
        top_neighbors = neighbors[sorted_idx[:5]]
        top_sims = similarities[sorted_idx[:5]]
        
        print(f"\n歌曲 {i}:")
        print(f"  鄰居數量: {row.nnz}")
        print(f"  前 5 個最相似的鄰居:")
        for j, (neighbor, sim) in enumerate(zip(top_neighbors, top_sims), 1):
            print(f"    {j}. 歌曲 {neighbor}: 相似度 = {sim:.4f}")
    else:
        print(f"\n歌曲 {i}: 沒有鄰居（異常！）")

# 檢查統計資訊
print("\n" + "=" * 60)
print("統計資訊")
print("=" * 60)
degrees = np.array(graph.sum(axis=1)).flatten()
print(f"平均度數: {degrees.mean():.2f}")
print(f"最小度數: {degrees.min()}")
print(f"最大度數: {degrees.max()}")
print(f"度數標準差: {degrees.std():.2f}")

# 檢查是否有孤立節點
isolated = (degrees == 0).sum()
print(f"\n孤立節點（沒有鄰居的歌曲）: {isolated}")

# 載入 metadata
print("\n" + "=" * 60)
print("Metadata 摘要")
print("=" * 60)
with open('outputs/graph/graph_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
    
print(f"Emotion 向量形狀: {metadata['emotion_vector_shape']}")
print(f"Topic 向量形狀: {metadata['topic_vector_shape']}")
print(f"合併向量形狀: {metadata['combined_vector_shape']}")
print(f"稀疏化策略: {metadata['sparsify_strategy']}")
print(f"Top-M: {metadata['top_m']}")
print(f"相似度範圍: [{metadata['similarity_range'][0]:.4f}, {metadata['similarity_range'][1]:.4f}]")
print(f"節點數: {metadata['num_nodes']}")
print(f"邊數: {metadata['num_edges']:,}")
print(f"稀疏度: {metadata['sparsity']*100:.2f}%")

print("\n" + "=" * 60)
print("✅ 檢查完成！")
print("=" * 60)


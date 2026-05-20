import pandas as pd
from ranx import Qrels, Run, evaluate

def calculate_all_metrics(qrels_path: str, run_path: str) -> dict:
    """
    Tính toán nDCG@1-10, Precision@1-10, và Recall@1-50 cho một file kết quả.
    """
    qrels_df = pd.read_parquet(qrels_path)
    run_df = pd.read_parquet(run_path)
    
    # SỬA TẠI ĐÂY: Ép hẳn sang kiểu object để vượt qua bộ check của ranx
    qrels_df["query_id"] = qrels_df["query_id"].astype(object)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(object)
    qrels_df["relevance"] = qrels_df["relevance"].astype(int)
    
    run_df["query_id"] = run_df["query_id"].astype(object)
    run_df["doc_id"] = run_df["doc_id"].astype(object)
    run_df["score"] = run_df["score"].astype(float)
    
    # Đóng gói vào đối tượng của Ranx
    qrels = Qrels.from_df(qrels_df, q_id_col="query_id", doc_id_col="doc_id", score_col="relevance")
    run = Run.from_df(run_df, q_id_col="query_id", doc_id_col="doc_id", score_col="score")
    
    # Tạo danh sách các mốc K theo yêu cầu
    metrics_list = []
    for k in range(1, 11):
        metrics_list.append(f"ndcg@{k}")
        metrics_list.append(f"precision@{k}")
        metrics_list.append(f"map@{k}")
        
    for k in range(1, 51):
        metrics_list.append(f"recall@{k}")
        
    # Tính toán tập trung
    scores = evaluate(qrels, run, metrics_list)
    return scores
import os
import pandas as pd
import logging
from src.evaluation.metrics import calculate_all_metrics
from src.utils.logging import get_logger
from src.config.paths import *

# Khởi tạo logger hệ thống từ cấu hình chung của bạn
logger = get_logger(__name__)

QRELS_PATH = "data/ground_truth/qrels.parquet"

METHODS = {
    "Dense": RUN_DENSE,
    "BM25": RUN_BM25,
    "Hybrid": RUN_HYBRID,
    "Rerank": RUN_RERANK
}

def main():
    if not os.path.exists(QRELS_PATH):
        logger.error("Không tìm thấy file Ground Truth tại %s", QRELS_PATH)
        return

    all_results = {}

    for name, run_path in METHODS.items():
        if not os.path.exists(run_path):
            logger.warning("Bỏ qua %s: Không tìm thấy file kết quả %s", name, run_path)
            continue
            
        logger.info("Đang tính toán metrics cho phương pháp: %s...", name)
        scores = calculate_all_metrics(QRELS_PATH, run_path)
        all_results[name] = scores

    if not all_results:
        logger.error("Không có dữ liệu kết quả nào được tính toán thành công.")
        return

    df_eval = pd.DataFrame(all_results).T
    df_eval = df_eval.reindex(sorted(df_eval.columns), axis=1)
    
    # Xuất toàn bộ báo cáo chi tiết bao gồm cả MAP ra file CSV cấu hình sẵn
    df_eval.to_csv(FULL_EVAL)
    logger.info("Đã xuất toàn bộ báo cáo chi tiết ra file %s", FULL_EVAL)

if __name__ == "__main__":
    main()
# Bộ dữ liệu Truy xuất bài báo ArXiv (2015–2025)

Liên kết đến dataset:
[Paper Retrieval Arxiv](https://huggingface.co/datasets/c1khoa/papers_retrieval_arxiv)

## Tổng quan

Bộ dữ liệu này được thiết kế cho **nghiên cứu truy xuất thông tin (IR) trên các bài báo khoa học**.
Nó bao gồm một bộ ngữ liệu lớn các bài báo được thu thập từ arXiv trong khoảng thời gian từ **2015 đến 2025**, cùng với một tập các câu truy vấn tổng hợp và nhãn đánh giá độ liên quan (qrels).

Bộ dữ liệu có thể được sử dụng để đánh giá các hệ thống truy xuất như:

- truy xuất từ khóa (BM25)
- truy xuất dày đặc (dense retrieval)
- mô hình xếp hạng lại bằng mạng nơ-ron (neural reranking)

Các tác vụ điển hình bao gồm:

- tìm kiếm ngữ nghĩa
- truy xuất văn bản
- truy xuất thông tin bằng mạng nơ-ron
- gợi ý bài báo học thuật

---

## Cấu trúc bộ dữ liệu

Bộ dữ liệu gồm ba tệp chính:

```
papers_2015_2025.parquet
queries.parquet
qrels.parquet
```

### 1. Ngữ liệu (Corpus)

`papers_2015_2025.parquet`

Tệp này chứa **ngữ liệu văn bản** gồm các bài báo từ arXiv.

| Trường      | Mô tả                       |
| ----------- | --------------------------- |
| id          | mã định danh arXiv          |
| title       | tiêu đề bài báo             |
| abstract    | tóm tắt bài báo             |
| authors     | danh sách tác giả           |
| categories  | chuyên mục chủ đề của arXiv |
| update_date | ngày cập nhật cuối          |

Ngữ liệu được lọc để chỉ bao gồm các bài báo thuộc các lĩnh vực sau:

- cs.AI
- cs.CL
- cs.IR
- cs.LG

và trong khoảng năm xuất bản **2015–2025**.

---

### 2. Câu truy vấn (Queries)

`queries.parquet`

Tệp này chứa các **câu truy vấn tìm kiếm** dùng để đánh giá truy xuất.

| Trường   | Mô tả                     |
| -------- | ------------------------- |
| query_id | mã định danh câu truy vấn |
| text     | nội dung câu truy vấn     |

Các câu truy vấn được tự động sinh ra dựa trên các mẫu chủ đề liên quan đến các lĩnh vực như:

- dịch máy bằng mạng nơ-ron
- truy xuất thông tin
- xếp hạng văn bản
- độ tương đồng ngữ nghĩa
- embedding mạng nơ-ron
- trả lời câu hỏi

Độ dài câu truy vấn dao động từ **1 đến 6 token**.

---

### 3. Nhãn đánh giá độ liên quan (Relevance Judgments)

`qrels.parquet`

Tệp này chứa **nhãn độ liên quan giữa câu truy vấn và văn bản**.

| Trường    | Mô tả                     |
| --------- | ------------------------- |
| query_id  | mã định danh câu truy vấn |
| doc_id    | mã định danh văn bản      |
| relevance | điểm độ liên quan         |

Điểm độ liên quan theo **thang đánh giá phân cấp**:

| Điểm | Ý nghĩa       |
| ---- | ------------- |
| 3    | rất liên quan |
| 2    | liên quan     |
| 1    | liên quan yếu |

Các nhãn được tạo ra bằng cách truy xuất kết quả từ API tìm kiếm của arXiv và lọc theo:

- năm xuất bản
- chuyên mục chủ đề
- sự tồn tại trong ngữ liệu cục bộ

---

## Mục đích sử dụng

Bộ dữ liệu này được thiết kế để đánh giá các hệ thống truy xuất bao gồm:

- BM25
- truy xuất bằng embedding dày đặc
- tìm kiếm láng giềng gần đúng (approximate nearest neighbor search)
- mô hình xếp hạng lại bằng mạng nơ-ron

Ví dụ về các tác vụ nghiên cứu:

- tìm kiếm ngữ nghĩa trên các bài báo khoa học
- truy xuất đoạn văn dày đặc (dense passage retrieval)
- truy xuất thông tin bằng mạng nơ-ron
- tìm kiếm tài liệu học thuật

---

## Ví dụ sử dụng

Tải bộ dữ liệu bằng Python:

```python
import pandas as pd

corpus = pd.read_parquet("papers_2015_2025.parquet")
queries = pd.read_parquet("queries.parquet")
qrels = pd.read_parquet("qrels.parquet")
```

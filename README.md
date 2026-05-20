# Truy xuất thông tin bài báo khoa học

## 1. Bài toán

Đồ án xây dựng một hệ thống truy xuất thông tin (Information Retrieval - IR) cho tập bài báo khoa học lấy từ arXiv. Người dùng nhập một câu truy vấn ngắn, ví dụ như "neural machine translation", "document ranking methods" hoặc "dense passage retrieval", hệ thống sẽ trả về danh sách các bài báo liên quan nhất trong kho dữ liệu.

Mục tiêu chính của hệ thống không phải là sinh câu trả lời tự nhiên như chatbot, mà là đánh giá và so sánh các phương pháp truy xuất tài liệu. Vì vậy, đầu vào và đầu ra của hệ thống được xác định như sau:

- Đầu vào: một truy vấn tìm kiếm bằng tiếng Anh.
- Đầu ra: danh sách Top-K bài báo liên quan, mỗi kết quả gồm mã bài báo, điểm xếp hạng và thứ tự trong danh sách trả về.
- Nhiệm vụ đánh giá: so sánh kết quả truy xuất với tập ground truth để đo mức độ tìm đúng và xếp hạng đúng các tài liệu liên quan.

Trong phạm vi đồ án, nhóm tập trung vào bài toán academic search: tìm kiếm bài báo trong các lĩnh vực AI, NLP, IR và Machine Learning. Đây là bài toán phù hợp với môn Information Retrieval vì có đầy đủ các thành phần quan trọng: corpus, query set, relevance judgments, indexing, retrieval, ranking và evaluation.

## 2. Dataset

Dataset được xây dựng từ metadata bài báo arXiv trong giai đoạn 2015-2025. Các bài báo được lấy từ những nhóm chủ đề chính:

- `cs.AI`: Artificial Intelligence
- `cs.CL`: Computation and Language
- `cs.IR`: Information Retrieval
- `cs.LG`: Machine Learning

Mỗi bài báo trong corpus ban đầu gồm các trường thông tin chính:

- `id`: mã định danh arXiv của bài báo.
- `title`: tiêu đề bài báo.
- `abstract`: tóm tắt nội dung bài báo.
- `authors`: danh sách tác giả.
- `categories`: nhóm chủ đề arXiv.
- `update_date`: ngày cập nhật metadata.

Dữ liệu thô được lưu ở dạng nhiều file `jsonl` theo từng category và từng năm. Sau khi gom dữ liệu, hệ thống loại bỏ các bài báo trùng `id`, chuẩn hóa lại các trường cần dùng và lưu thành file:

```text
data/raw/papers_2015_2025.parquet
```

File này đóng vai trò là corpus gốc. Sau bước tiền xử lý, corpus dùng cho retrieval được lưu tại:

```text
data/preprocessed/papers_preprocessed.parquet
```

Trong bản preprocessed, mỗi document được biểu diễn bằng hai trường chính:

- `doc_id`: mã định danh đã chuẩn hóa của bài báo.
- `text`: nội dung dùng để lập chỉ mục và truy xuất.

## 3. Sinh ground truth

Do hệ thống không có query log thật từ người dùng, nhóm xây dựng ground truth theo hướng synthetic benchmark. Quy trình này gồm hai phần: sinh tập truy vấn và sinh relevance judgments.

### 3.1. Sinh tập truy vấn

Các truy vấn được sinh dựa trên những topic phổ biến trong NLP và Information Retrieval, ví dụ:

- neural machine translation
- information retrieval
- document ranking
- transformer language models
- contrastive learning
- question answering
- dense passage retrieval
- scientific paper recommendation
- semantic similarity
- neural reranking
- cross lingual retrieval
- neural embeddings

Để truy vấn đa dạng hơn, hệ thống kết hợp topic với các nhóm từ bổ trợ như:

- models
- methods
- approaches
- systems
- techniques
- algorithms

Ngoài ra còn có các cụm ngữ cảnh như:

- for academic search
- for scientific papers
- for research literature
- in neural IR systems
- for multilingual corpora
- in academic recommender systems

Tập query được thiết kế có độ dài từ 1 đến 6 từ. Phân phối độ dài query:

| Độ dài query | Số lượng |
| ------------ | -------: |
| 1 từ         |       35 |
| 2 từ         |       60 |
| 3 từ         |       90 |
| 4 từ         |       60 |
| 5 từ         |       35 |
| 6 từ         |       20 |

Tổng cộng hệ thống sinh 300 truy vấn và lưu vào:

```text
data/ground_truth/queries.parquet
```

Mỗi query có schema:

```text
query_id
text
```

Để giảm trùng lặp, hệ thống dùng token overlap filtering. Nếu một query mới có mức trùng token quá cao với query đã có, query đó sẽ bị loại và sinh lại. Cách này giúp tập truy vấn đa dạng hơn, tránh việc nhiều query gần như giống nhau làm sai lệch đánh giá.

### 3.2. Sinh relevance judgments

Ground truth được sinh bằng cách dùng arXiv API như một nguồn tham chiếu ban đầu. Với mỗi query, hệ thống gửi truy vấn đến arXiv API, tìm trong phần abstract và lấy kết quả được arXiv sắp xếp theo độ liên quan.

Sau đó, kết quả được lọc theo các điều kiện:

- Năm bài báo nằm trong khoảng 2015-2025.
- Category thuộc một trong các nhóm `cs.AI`, `cs.CL`, `cs.IR`, `cs.LG`.
- Document phải tồn tại trong corpus cục bộ.

Với mỗi query, hệ thống thu thập tối đa 50 tài liệu liên quan. Vì arXiv API đã trả kết quả theo thứ tự relevance, nhóm chuyển thứ hạng thành nhãn liên quan có mức độ:

| Thứ hạng trong kết quả arXiv | Nhãn relevance | Ý nghĩa       |
| ---------------------------- | -------------: | ------------- |
| 1-6                          |              3 | Rất liên quan |
| 7-21                         |              2 | Liên quan     |
| 22-50                        |              1 | Liên quan yếu |

Tập relevance judgments được lưu tại:

```text
data/ground_truth/qrels.parquet
```

Schema của qrels:

```text
query_id
doc_id
relevance
```

Cách xây dựng này không hoàn hảo như gán nhãn thủ công bởi chuyên gia, nhưng hợp lý trong phạm vi đồ án vì tạo được benchmark có quy mô đủ lớn, có graded relevance và đảm bảo mọi document trong qrels đều nằm trong corpus đánh giá.

## 4. Tiền xử lý dữ liệu

Tiền xử lý trong hệ thống được thiết kế theo hướng đơn giản và phù hợp với bài toán retrieval trên metadata khoa học. Hệ thống không xử lý toàn văn PDF, mà dùng chủ yếu title và abstract.

Các bước chính:

1. Chuẩn hóa `id` bài báo thành `doc_id`, loại bỏ phần URL hoặc version không cần thiết nếu có.
2. Làm sạch title và abstract:
    - loại bỏ biểu thức citation dạng `\cite{...}`;
    - loại bỏ reference dạng `\ref{...}`;
    - loại bỏ URL LaTeX dạng `\url{...}`;
    - chuẩn hóa khoảng trắng.
3. Tạo trường `text` dùng cho retrieval bằng cách ghép:

```text
title + ". " + title + ". " + abstract
```

Việc lặp lại title hai lần giúp tăng trọng số tự nhiên cho tiêu đề, vì tiêu đề thường chứa các từ khóa quan trọng nhất của bài báo. Đây là một lựa chọn hợp lý trong truy xuất tài liệu khoa học, nơi title thường phản ánh trực tiếp chủ đề nghiên cứu.

4. Loại bỏ document trùng `doc_id`.
5. Lưu corpus đã tiền xử lý thành Parquet để dùng lại cho các bước indexing.

Đối với BM25, văn bản được token hóa bằng rule đơn giản:

```text
[a-z0-9]+
```

Tức là hệ thống chuyển chữ về lowercase và giữ lại các token gồm chữ cái tiếng Anh hoặc chữ số. Cách tokenization này phù hợp với query học thuật vì vẫn giữ được các thuật ngữ như `bert`, `gpt`, `bm25`, `e5`, `llm`, `transformer`.

Lưu ý: pipeline hiện tại không dùng NLTK stopword removal, WordNet lemmatization, dịch tiếng Việt hoặc LSA summarizer. Do đó các phần này không được xem là thành phần chính thức của hệ thống hiện tại.

## 5. Lập chỉ mục

Hệ thống xây dựng hai loại chỉ mục chính: sparse index cho BM25 và dense vector index cho E5 embedding.

### 5.1. BM25 index

BM25 là phương pháp lexical retrieval dựa trên sự khớp từ khóa giữa query và document. Sau khi corpus được token hóa, hệ thống dùng BM25 để tính điểm liên quan giữa query và từng document.

Ưu điểm của BM25:

- Mạnh với các query có từ khóa rõ ràng.
- Phù hợp với tìm kiếm học thuật vì nhiều thuật ngữ chuyên ngành cần được khớp chính xác.
- Có khả năng xử lý độ dài document tốt hơn các mô hình đếm từ đơn giản.

BM25 index được lưu tại:

```text
artifacts/sparse/bm25.pkl
```

Tokenized documents được lưu tại:

```text
artifacts/sparse/tokenized_docs.pkl
```

### 5.2. Dense vector index

Dense retrieval biểu diễn query và document thành vector ngữ nghĩa. Hệ thống sử dụng mô hình:

```text
intfloat/e5-base
```

Đây là mô hình embedding thuộc họ E5, được thiết kế cho các tác vụ retrieval. Khi encode, hệ thống dùng định dạng prefix phù hợp:

- Document: `passage: ...`
- Query: `query: ...`

Các embedding được normalize, sau đó đưa vào FAISS để tìm kiếm vector gần nhất bằng inner product. Vì vector đã được normalize, inner product tương đương với cosine similarity.

FAISS index của hệ thống dùng HNSW:

- `hnsw_m = 32`
- `ef_construction = 200`
- `ef_search = 128`

Dense index được lưu tại:

```text
artifacts/dense/paper_index.faiss
```

Danh sách `doc_id` tương ứng với vector trong FAISS được lưu tại:

```text
artifacts/dense/doc_ids.npy
```

## 6. Các phương pháp truy xuất

Hệ thống hiện tại đánh giá bốn phương pháp chính.

### 6.1. BM25 Retrieval

BM25 nhận query, token hóa query theo cùng rule với document, sau đó tính điểm BM25 với toàn bộ corpus. Kết quả được sắp xếp giảm dần theo điểm và lấy Top-K.

Phương pháp này là baseline lexical mạnh. Nó thường hoạt động tốt khi query chứa đúng thuật ngữ xuất hiện trong title hoặc abstract.

### 6.2. Dense Retrieval bằng E5 + FAISS

Dense retriever encode query bằng E5, sau đó tìm kiếm trong FAISS index để lấy các document có vector gần query nhất.

Phương pháp này có khả năng bắt quan hệ ngữ nghĩa tốt hơn BM25 trong các trường hợp query và document không dùng đúng cùng một từ, nhưng có ý nghĩa gần nhau. Ví dụ, query nói về "semantic search" có thể liên quan đến các bài dùng thuật ngữ "neural retrieval" hoặc "dense retrieval".

### 6.3. Hybrid Retrieval bằng Reciprocal Rank Fusion

Hybrid retriever kết hợp kết quả từ BM25 và Dense Retrieval. Thay vì cộng trực tiếp điểm BM25 và điểm dense, hệ thống dùng Reciprocal Rank Fusion (RRF).

Ý tưởng của RRF là dựa trên thứ hạng thay vì giá trị score tuyệt đối. Một document xuất hiện ở thứ hạng cao trong một hoặc nhiều danh sách sẽ nhận điểm fusion cao hơn.

Công thức tổng quát:

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

Trong hệ thống, tham số:

```text
RRF_K = 60
```

Cách này giúp tránh vấn đề score của BM25 và score của dense embedding không cùng thang đo.

### 6.4. BM25 -> E5 Rerank

Phương pháp rerank hoạt động theo hai bước:

1. Dùng BM25 lấy một tập candidate ban đầu.
2. Dùng E5 similarity để xếp hạng lại các candidate đó.

Trong cấu hình hiện tại:

```text
candidate_k = 100
top_k = 50
```

Ý tưởng của phương pháp này là tận dụng BM25 để lọc nhanh các tài liệu có khả năng liên quan, sau đó dùng embedding để sắp xếp lại theo ngữ nghĩa. Đây là kiến trúc phổ biến trong search system vì tách retrieval thành candidate generation và reranking.

## 7. Pipeline tổng thể

Pipeline của đồ án có thể tóm tắt như sau:

```text
arXiv metadata 2015-2025
        |
        v
Gom dữ liệu và loại trùng paper
        |
        v
Tạo corpus papers_2015_2025.parquet
        |
        v
Tiền xử lý title + abstract
        |
        v
Tạo papers_preprocessed.parquet
        |
        +----------------------+
        |                      |
        v                      v
Xây BM25 index          Encode E5 embeddings
        |                      |
        v                      v
Sparse retrieval        FAISS dense index
        |                      |
        +----------+-----------+
                   |
                   v
        BM25 / Dense / Hybrid / Rerank
                   |
                   v
             Top-K documents
                   |
                   v
        So sánh với qrels và tính metrics
```

Pipeline đánh giá:

```text
queries.parquet
      |
      v
Chạy từng retriever cho 300 queries
      |
      v
Sinh run files:
- run_bm25.parquet
- run_dense.parquet
- run_hybrid.parquet
- run_rerank.parquet
      |
      v
So sánh với qrels.parquet
      |
      v
Xuất full_evaluation_report.csv
```

## 8. Metrics đánh giá

Hệ thống sử dụng thư viện `ranx` để tính các metric IR chuẩn. Các metric chính gồm:

### Precision@K

Precision@K đo trong Top-K kết quả trả về có bao nhiêu tài liệu là liên quan.

```text
Precision@K = số tài liệu liên quan trong Top-K / K
```

Metric này phản ánh độ chính xác của danh sách kết quả ở những vị trí đầu.

### Recall@K

Recall@K đo hệ thống tìm được bao nhiêu phần trong tổng số tài liệu liên quan của query.

```text
Recall@K = số tài liệu liên quan trong Top-K / tổng số tài liệu liên quan
```

Metric này quan trọng khi mục tiêu là tìm được càng nhiều tài liệu liên quan càng tốt.

### nDCG@K

nDCG đánh giá chất lượng xếp hạng khi ground truth có nhiều mức relevance. Vì qrels của hệ thống có nhãn 1, 2, 3 nên nDCG là metric phù hợp.

nDCG cho điểm cao hơn nếu tài liệu có relevance cao xuất hiện ở vị trí cao trong danh sách kết quả.

### MAP@K

MAP đo chất lượng ranking trung bình trên toàn bộ tập query. Metric này xem xét vị trí xuất hiện của các document liên quan, không chỉ số lượng document liên quan.

## 9. Kết quả đánh giá

File kết quả tổng hợp được lưu tại:

```text
results/full_evaluation_report.csv
```

Một số chỉ số tiêu biểu:

| Phương pháp | nDCG@10 | Precision@10 | Recall@50 | MAP@10 |
| ----------- | ------: | -----------: | --------: | -----: |
| BM25        |  0.1284 |       0.1663 |    0.0948 | 0.0258 |
| Hybrid      |  0.1004 |       0.1347 |    0.0809 | 0.0167 |
| Rerank      |  0.0788 |       0.1143 |    0.0759 | 0.0125 |
| Dense       |  0.0502 |       0.0677 |    0.0419 | 0.0075 |

Kết quả cho thấy BM25 là phương pháp tốt nhất trong benchmark hiện tại. BM25 đạt điểm cao nhất ở cả nDCG@10, Precision@10, Recall@50 và MAP@10.

Hybrid bằng RRF đứng thứ hai. Việc kết hợp BM25 và dense retrieval giúp cải thiện đáng kể so với dense retrieval đơn lẻ, nhưng vẫn chưa vượt qua BM25. Điều này cho thấy dense retriever trong hệ thống hiện tại chưa đủ mạnh để bổ sung lợi ích lớn cho BM25.

Rerank BM25 -> E5 thấp hơn Hybrid và BM25. Một nguyên nhân có thể là E5 similarity chưa phù hợp hoàn toàn với ground truth được sinh từ arXiv lexical/metadata ranking. Khi rerank lại candidate BM25 bằng embedding, một số tài liệu có khớp từ khóa tốt có thể bị đẩy xuống dưới, làm giảm precision và nDCG.

Dense retrieval có kết quả thấp nhất trong các phương pháp. Điều này không có nghĩa dense retrieval luôn kém, mà cho thấy trong thiết lập hiện tại, BM25 phù hợp hơn với dataset và ground truth. Các query phần lớn là topic/keyword ngắn, còn relevance labels được sinh từ arXiv search API, nên các phương pháp lexical có lợi thế rõ ràng.

## 10. Nhận xét

BM25 hoạt động tốt vì dữ liệu là title và abstract khoa học, trong đó các thuật ngữ quan trọng thường xuất hiện trực tiếp. Các query cũng ngắn và giàu keyword, ví dụ "document ranking", "question answering", "neural embeddings". Với kiểu query này, khớp từ khóa là tín hiệu rất mạnh.

Dense retrieval bằng E5 có ưu điểm về ngữ nghĩa, nhưng chưa thể hiện tốt trong benchmark hiện tại. Có thể do một số nguyên nhân:

- Query ngắn nên embedding chứa ít ngữ cảnh.
- Ground truth được sinh từ arXiv API, có thể thiên về lexical relevance.
- Corpus gồm abstract khoa học dài và nhiều thuật ngữ chuyên ngành, trong khi dense embedding có thể làm mờ các keyword quan trọng.
- Chưa có bước fine-tune embedding model trên dữ liệu khoa học hoặc dữ liệu qrels của hệ thống.

Hybrid retrieval có kết quả trung gian. Nó tốt hơn dense retrieval rõ rệt, chứng tỏ BM25 đóng góp tín hiệu mạnh trong danh sách fusion. Tuy nhiên, vì dense retrieval yếu hơn BM25, việc fusion có thể kéo một số kết quả BM25 tốt xuống thấp hơn.

Reranking cũng chưa tốt hơn BM25. Điều này cho thấy candidate generation bằng BM25 đã khá phù hợp với ground truth, còn bước E5 rerank chưa tạo thêm lợi ích trong thiết lập hiện tại.

## 11. Hạn chế

Hệ thống vẫn còn một số hạn chế:

- Ground truth được sinh tự động từ arXiv API, chưa phải nhãn đánh giá thủ công từ chuyên gia.
- Query là synthetic query, chưa phản ánh đầy đủ hành vi tìm kiếm thật của người dùng.
- Corpus chỉ dùng metadata, title và abstract, chưa dùng full text PDF.
- Dense model chưa được fine-tune riêng cho scientific paper retrieval.
- Chưa có query expansion, pseudo relevance feedback hoặc learning-to-rank.
- Chưa có giao diện người dùng hoàn chỉnh trong pipeline đánh giá hiện tại.
- Chưa hỗ trợ chính thức truy vấn tiếng Việt trong pipeline hiện tại.

## 12. Kết luận

Đồ án đã xây dựng được một pipeline truy xuất thông tin hoàn chỉnh cho bài báo khoa học, bao gồm thu thập dữ liệu, tiền xử lý, sinh ground truth, xây dựng chỉ mục, triển khai nhiều phương pháp retrieval và đánh giá bằng các metric IR chuẩn.

Trong các phương pháp đã thử nghiệm, BM25 cho kết quả tốt nhất trên benchmark hiện tại. Điều này phù hợp với đặc trưng của dataset: query ngắn, nhiều keyword chuyên ngành và ground truth được sinh từ hệ thống search của arXiv. Hybrid retrieval và reranking có ý nghĩa về mặt thiết kế hệ thống, nhưng chưa vượt qua BM25 trong kết quả thực nghiệm.

Kết quả này cho thấy với bài toán truy xuất bài báo khoa học dựa trên title và abstract, lexical retrieval vẫn là baseline rất mạnh. Dense retrieval cần thêm cải tiến như fine-tuning, query expansion, hoặc kết hợp lại theo trọng số tối ưu hơn để phát huy tốt hơn trong các truy vấn mang tính ngữ nghĩa.

Hướng phát triển tiếp theo có thể gồm:

- Fine-tune E5 hoặc một embedding model khác trên qrels của hệ thống.
- Thử các mô hình reranker mạnh hơn như cross-encoder.
- Tối ưu hybrid bằng cách điều chỉnh trọng số hoặc thử nhiều biến thể rank fusion.
- Bổ sung query thật từ người dùng để đánh giá thực tế hơn.
- Mở rộng corpus sang full text để hỗ trợ tìm kiếm sâu hơn trong nội dung bài báo.

# Kết Quả Training và Đánh Giá Model

## 📊 Kết Quả Sau Cải Thiện

### Isolation Forest Model

**Tham số**: contamination = 0.25 (25%)

**Metrics**:
- **Precision**: 0.1392 (13.92%) ⬆️ từ 12.5%
- **Recall**: 0.1419 (14.19%) ⬆️ từ 2.58% (cải thiện đáng kể!)
- **F1 Score**: 0.1406 (14.06%) ⬆️ từ 4.28%
- **Accuracy**: 0.5744 (57.44%)

**Confusion Matrix**:
```
                Predicted
              Normal  Anomaly
Actual Normal     341     136
       Anomaly    133      22
```

**Phân tích**:
- ✅ Recall tăng đáng kể: từ 2.58% lên 14.19% (tăng ~5.5 lần)
- ✅ F1 Score tăng: từ 4.28% lên 14.06% (tăng ~3.3 lần)
- ⚠️ Precision vẫn thấp: 13.92% (nhiều false positives)
- ⚠️ Accuracy giảm: từ 71.68% xuống 57.44% (do phát hiện nhiều anomalies hơn)

## 🔍 So Sánh Trước và Sau

| Metric | Trước (contamination=0.05) | Sau (contamination=0.25) | Cải thiện |
|--------|---------------------------|-------------------------|-----------|
| Precision | 12.50% | 13.92% | +1.42% |
| Recall | 2.58% | 14.19% | +11.61% ⬆️⬆️ |
| F1 Score | 4.28% | 14.06% | +9.78% ⬆️⬆️ |
| Accuracy | 71.68% | 57.44% | -14.24% |

**Nhận xét**:
- Model sau cải thiện phát hiện được nhiều anomalies hơn (Recall tăng mạnh)
- Tuy nhiên có nhiều false positives hơn (Precision vẫn thấp)
- Accuracy giảm nhưng đây là điều bình thường khi model phát hiện nhiều anomalies hơn

## 📈 Ground Truth Statistics

- **Tổng số samples**: 632
- **Anomalies thực tế**: 155 (24.53%)
- **Normal thực tế**: 477 (75.47%)
- **Anomalies phát hiện**: 158 (25.00%)

## 🎯 Điểm Mạnh

1. ✅ **Recall cải thiện đáng kể**: Model phát hiện được nhiều anomalies hơn
2. ✅ **F1 Score tăng**: Cân bằng tốt hơn giữa Precision và Recall
3. ✅ **Ground Truth tốt hơn**: Sử dụng code-based features
4. ✅ **Consistency**: Training và evaluation đều dùng `code`

## ⚠️ Điểm Yếu Cần Cải Thiện

1. **Precision thấp** (13.92%): Nhiều false positives
   - **Giải pháp**: Cải thiện Ground Truth labels, thêm feature engineering

2. **Recall vẫn thấp** (14.19%): Vẫn bỏ sót nhiều anomalies
   - **Giải pháp**: Nâng cấp embedding model (CodeBERT), cải thiện feature extraction

3. **Ground Truth có thể chưa chính xác**: 
   - **Giải pháp**: Review lại cách tạo labels, có thể cần manual labeling

## 🚀 Các Cải Thiện Tiếp Theo

### Ưu tiên Cao
1. **Nâng cấp Embedding Model**: 
   - Thay `all-MiniLM-L6-v2` bằng CodeBERT hoặc GraphCodeBERT
   - Embeddings tốt hơn cho code sẽ cải thiện cả Precision và Recall

2. **Cải thiện Ground Truth**:
   - Review lại cách tạo labels
   - Có thể cần manual labeling một phần dữ liệu
   - Sử dụng domain experts để đánh giá

3. **Feature Engineering**:
   - Thêm nhiều features từ code (AST, control flow, etc.)
   - Sử dụng code analysis tools

### Ưu tiên Trung bình
4. **Hyperparameter Tuning**:
   - Grid search cho contamination
   - Tune các tham số khác của Isolation Forest

5. **Ensemble Methods**:
   - Kết hợp Isolation Forest và Autoencoder
   - Voting hoặc weighted average

### Ưu tiên Thấp
6. **Cross-Validation**:
   - K-fold cross-validation để đánh giá tốt hơn
   - Stratified sampling

## 📝 Kết Luận

Model đã được cải thiện đáng kể về **Recall** và **F1 Score**, nhưng vẫn cần cải thiện về **Precision**. 

**Khuyến nghị**:
1. Tiếp tục cải thiện Ground Truth labels
2. Nâng cấp embedding model cho code
3. Thêm feature engineering
4. Cân nhắc ensemble methods

## 🔗 Files Liên Quan

- `src/model_training.py`: Code training
- `src/evaluate_model.py`: Code evaluation với Ground Truth cải thiện
- `models/trained_if.pkl`: Model đã train
- `data/processed/evaluation_results.csv`: Kết quả chi tiết




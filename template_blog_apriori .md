
📦Case Study: Nâng cấp Mô hình Dự báo Ô nhiễm - Từ ARIMA đến SARIMA & Tính Mùa Vụ
👥 Thông tin Nhóm
Nhóm: FIT DNU CONQUER

Thành viên:

Vũ Bích Hợp
Đỗ Quang Huy
Lâm Xuân Trường
...

Chủ đề: Chủ đề 5.3.2 - SARIMA – Thêm yếu tố mùa vụ (Seasonality).

Dataset: Beijing Air Quality (Dữ liệu chuỗi thời gian).

## Mục tiêu
Mục tiêu của nhóm là:

Không chỉ dự báo nồng độ bụi tăng hay giảm chung chung, mà là "bắt mạch" được nhịp thở của thành phố. Chúng tôi muốn chứng minh ô nhiễm không khí tuân theo một chu kỳ lặp lại chính xác từng giờ (24h/ngày), từ đó xây dựng hệ thống cảnh báo sớm thông minh hơn thay vì chỉ nhìn vào quá khứ một cách máy móc.

## 1. Ý tưởng & Feynman Style
ARIMA (Cũ) là gì?Hãy tưởng tượng bạn lái xe và chỉ nhìn vào gương chiếu hậu. Nếu 5 phút trước đường đông, bạn đoán 5 phút tới đường cũng đông. ARIMA làm việc như vậy, nó dự báo tương lai dựa trên xu hướng ngay sát đó. Nhưng nó "mù" với các quy luật lặp lại (như giờ cao điểm).

Tại sao cần SARIMA (Chủ đề 5.3.2)?Ô nhiễm không khí giống như kẹt xe: Sáng nào 7h cũng tắc, tối nào 18h cũng đông. Đó gọi là Mùa vụ (Seasonality).SARIMA thông minh hơn ARIMA ở chỗ nó có "bộ nhớ chu kỳ". Nó sẽ bảo: "Này, giờ này hôm qua bụi cao lắm đấy, khả năng hôm nay cũng thế, đừng chỉ nhìn vào 1 tiếng trước!".

Ý tưởng thuật toán (1-2 câu):Cấu hình "nhịp tim" $s=24$ (chu kỳ 24 giờ) vào mô hình để máy tính học được thói quen sinh hoạt của cả thành phố.

## 2. Quy trình Thực hiện

1) Load & Resample: Chuyển dữ liệu về dạng chuỗi thời gian liên tục, làm đều dữ liệu theo từng giờ (Hourly).

2) Khám sức khỏe dữ liệu (ACF): Dùng biểu đồ tự tương quan để tìm "nhịp tim" (chu kỳ lặp lại).

3) Xây dựng ARIMA (Baseline): Chạy mô hình cơ bản để làm mốc so sánh.

4) Xây dựng SARIMA (Challenger): Thêm tham số seasonal_order=(P,D,Q,24) để bắt chu kỳ ngày.

5) Huấn luyện & Đối đầu: Cho 2 mô hình chạy đua trên tập dữ liệu Test (7 ngày cuối).

6) Đánh giá: So sánh bằng AIC (Độ phức tạp) và RMSE (Độ sai lệch).

7) Đề xuất: Chiến lược quản lý môi trường dựa trên khung giờ dự báo.

## 3. Tiền xử lý Dữ liệu
- Những bước làm sạch:
Chuyển cột datetime sang index chuẩn.
Resample 'H': Gom nhóm dữ liệu theo giờ để đảm bảo tính liên tục (không bị đứt quãng).
Interpolate: Nội suy tuyến tính để lấp đầy các khoảng trống dữ liệu bị thiếu (Missing values).
- Thống kê nhanh:
Quãng thời gian: 2013 - 2017.
Tổng số điểm dữ liệu huấn luyện: ~34,000 giờ.
Tập kiểm thử (Test set): 168 giờ cuối (1 tuần).
## 4. Áp dụng SARIMA
Tham số sử dụng:
Seasonality (s): 24 (Tương ứng với chu kỳ 1 ngày).

Chiến lược: So sánh trực tiếp hiệu quả cải thiện giữa Non-seasonal vs Seasonal.
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. ARIMA (Mô hình cũ - Baseline)
# Chỉ nhìn thấy xu hướng ngắn hạn
model_arima = ARIMA(y_train, order=(1,1,1))
res_arima = model_arima.fit()

# 2. SARIMA (Mô hình mới - Có mùa vụ)
# Thêm seasonal_order=(1,0,1,24) để bắt quy luật 24h
model_sarima = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,0,1,24))
res_sarima = model_sarima.fit(disp=False)

# Kết quả: SARIMA có AIC thấp hơn đáng kể -> Tốt hơn
print(f"ARIMA AIC: {res_arima.aic} vs SARIMA AIC: {res_sarima.aic}")

## 5. Trực quan hóa (Visualization)
- Hình 1: Biểu đồ ACF - Bằng chứng thép
Mô tả: Các cột dựng đứng cao vút ở mốc Lag 24, Lag 48, Lag 72... trông như chiếc lược. Điều này chứng minh nồng độ bụi lặp lại y hệt sau mỗi 24h. Đây là cơ sở để chọn SARIMA.

- Hình 2: Cuộc đối đầu (Forecast Battle)
Mô tả: Đường màu đỏ (ARIMA) đi khá phẳng và "cứng". Đường màu xanh (SARIMA) uốn lượn nhịp nhàng, bám sát các đỉnh nhọn (Peak) của dữ liệu thực tế (Màu đen).

## 6. Insight từ Kết quả
Insight #1 (Chu kỳ sinh hoạt): Ô nhiễm PM2.5 tại Bắc Kinh tuân thủ chặt chẽ quy luật 24h. Đỉnh ô nhiễm thường rơi vào khung giờ giao thông cao điểm hoặc ban đêm (do tích tụ khí quyển), và giảm sâu vào giữa trưa.

Insight #2 (Sức mạnh của Lag 24): Việc biết nồng độ bụi của "giờ này ngày hôm qua" quan trọng hơn nhiều so với việc biết nồng độ của "1 tiếng trước". SARIMA đã tận dụng điều này để giảm chỉ số lỗi (RMSE) xuống đáng kể.

Insight #3 (Cảnh báo sai của ARIMA): Mô hình ARIMA thường dự báo trễ pha (Lag) - tức là khi bụi tăng rồi nó mới dự báo tăng. SARIMA nhờ hiểu quy luật nên có thể dự báo tăng trước khi nó thực sự xảy ra.

Insight #4: Vào các ngày cuối tuần, biên độ dao động có sự thay đổi nhẹ so với ngày thường, gợi ý nên mở rộng nghiên cứu thêm chu kỳ tuần ($s=168$) nếu tài nguyên máy tính cho phép.

## 7. Kết luận & Đề xuất Kinh doanh
Dựa trên kết quả mô hình, nhóm đề xuất các chiến lược hành động:

Hệ thống Cảnh báo sớm (Early Warning):

Gửi tin nhắn cảnh báo người dân vào 2 khung giờ cố định được dự báo là đỉnh ô nhiễm (ví dụ: 7h sáng và 9h tối) để họ chủ động đeo khẩu trang.

Điều tiết Giao thông thông minh:

Dựa vào biểu đồ chu kỳ ngày, áp dụng hạn chế xe tải hạng nặng vào các khung giờ mà mô hình dự báo nồng độ bụi sẽ vượt ngưỡng nguy hại.

Thanh tra môi trường:

Nếu mô hình dự báo đêm nay bụi thấp (theo quy luật), nhưng thực tế bụi lại tăng vọt -> Khả năng cao có nhà máy xả thải trộm. Cần cử đội thanh tra đi kiểm tra ngay lập tức (Anomaly Detection).


## 8. Link Code & Notebook


## 9. Slide trình bày
- Link Slide:



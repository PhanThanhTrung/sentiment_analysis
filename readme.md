# Sentiment Analysis
Sentiment analysis là một trong những bài toán nổi bật trong lĩnh vực xử lý ngôn ngữ tự nhiên. Có nhiều cách tiếp cận bài toán, mỗi phương pháp đều có điểm mạnh và điểm yếu riêng. Trong repo cung cấp code train, test và evaluate theo 2 cách khác nhau đối với bài toán sentiment analysis. 

Cấu trúc của repo:
```.
├── Sentiment_CNN
│   ├── embedding_model.py
│   ├── evaluate.py
│   ├── models.h5
│   ├── predict.py
│   ├── train.py
│   └── word_embedding.model
├── Sentiment_LSTM
│   ├── evaluate.py
│   ├── models.h5
│   ├── predict.py
│   ├── tokenizer.pickle
│   └── train.py
├── readme.md
└── requirement.txt
```
1. repo cung cấp code training sentiement analysis cho 2 phương pháp tiếp cận: CNN và LSTM. Để có thể training sentiment. Thay đổi đường dẫn tới data của bạn ở biến `data_path`. Sau đó bạn có thể start training bằng cách:
```
    python3 train.py
```
2. để evaluate model trên tập test của data. Thay đổi đường dẫn tới data test ở biến `data_path`. Sau đó bạn có thể evaluate kết quả của model bằng cách run:
```
    python3 evaluate.py
```
3. để test model với trained weight. Thay đổi biến `text` thành câu mà bạn muốn test thử. Sau đó bạn có thể test model bằng cách run:
```
    python3 predict.py
```

Happy coding! 😎
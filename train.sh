python3 train.py \
    --embedding_model_path '/Users/hit.fluoxetine/HIT/repo/sentiment_analysis/models/word_embeddings/wiki.vi.model.bin' \
    --data_path '/Users/hit.fluoxetine/Dataset/nlp/data_train'\
    --log_dir /root/logs \
    --epochs 100 \
    --num_classes 2\
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_workers 0 \
    --log_iter 1000\

python3 train.py \
    --embedding_model_path 'models/word_embeddings/sa_embedding_model.bin' \
    --data_path '/Users/hit.fluoxetine/Dataset/nlp/data_train'\
    --log_dir ./logs \
    --epochs 10 \
    --num_classes 2\
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 8 \
    --log_iter 100\

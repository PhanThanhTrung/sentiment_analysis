import argparse
import logging
import os

import gensim
import gensim.models.keyedvectors as word2vec
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from clearml import Task
from CocCocTokenizer import PyTokenizer
from gensim.models import KeyedVectors, Word2Vec
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_processing.dataset import DatasetLoader
from loss.focal_loss import FocalLoss
from lstm import LSTM

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


task = Task.init(project_name='sentiment_analysis',
                 task_name='25042021_LSTM_bidirectional_2layer_word2vec')


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    os.makedirs(args.log_dir, exist_ok=True)

    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": lambda x: [[elem[0] for elem in x], [elem[1] for elem in x]]
    }

    # initiate a train generator
    train_data_path = os.path.join(args.data_path, 'train')
    train_loader = DatasetLoader(train_data_path, args.embedding_model_path)
    train_generator = DataLoader(train_loader, **params)

    # initiate a val generator
    val_data_path = os.path.join(args.data_path, 'test')
    val_loader = DatasetLoader(val_data_path, args.embedding_model_path)
    val_generator = DataLoader(val_loader, **params)
    vocab_size = train_loader.vocab_size

    # initiate a writer
    writer = SummaryWriter()

    # initialize a criterion
    criterion = FocalLoss()

    # initiate model
    model = LSTM(vocab_size=vocab_size, embedding_dim=400, hidden_dim=2000,
                 num_classes=args.num_classes, n_layers=2, bidirectional=True, dropout=0.5)

    # initialize an optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # initialize a learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_generator))

    train_loss = 0
    train_accuracy = 0
    log_iter_loss = 0
    log_iter_acc = 0
    index = 0
    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch))
        for comments, labels in tqdm.tqdm(train_generator, total=len(train_generator), desc='Training'):
            labels = torch.tensor(labels)
            if len(labels.size()) == 0:
                labels = torch.tensor([labels]).to(device)

            text_lengths = list(map(len, comments))
            padded = torch.nn.utils.rnn.pad_sequence(
                comments, batch_first=True)
            text_lengths = torch.tensor(text_lengths)
            optimizer.zero_grad()

            output = model(padded, text_lengths)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            train_loss += loss.item()
            log_iter_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(
                equals.type(torch.FloatTensor)).item()
            log_iter_acc += torch.mean(
                equals.type(torch.FloatTensor)).item()

            if index % args.log_iter == 0 and index != 0:
                test_loss = 0
                test_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for comments, labels in tqdm.tqdm(val_generator, total=len(val_generator), desc='Validation'):
                        labels = torch.tensor(labels)
                        if len(labels.size()) == 0:
                            labels = torch.tensor([labels]).to(device)

                        text_lengths = list(map(len, comments))
                        padded = torch.nn.utils.rnn.pad_sequence(
                            comments, batch_first=True)
                        text_lengths = torch.tensor(text_lengths)

                        out_val = model(padded, text_lengths)
                        batch_loss = criterion(output, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(out_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        test_accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()
                model.train()

                # Tensorboard log
                writer.add_scalar("Train_Accuracy",
                                  train_mean_acc, index)
                writer.add_scalar("Validation_Accuracy",
                                  val_mean_acc, index)
                writer.add_scalar(
                    "Train_loss", mean_train_loss, index)
                writer.add_scalar(
                    "Validation_loss", mean_test_loss, index)

                # Reset values
                log_iter_loss = 0
                log_iter_acc = 0

                # save model
                state_dict_name = "{}_{}_{:.2f}_{:.2f}.pth".format(
                    epoch, index, mean_test_loss, val_mean_acc)
                path_model_state_dict = os.path.join(
                    args.log_dir, 'state_dict')
                path_model_state_dict = os.path.join(
                    path_model_state_dict, state_dict_name)
                path_model = os.path.join(args.log_dir, 'model')
                path_model = os.path.join(path_model, state_dict_name)
                torch.save(model.state_dict(), path_model_state_dict)
                torch.save(model, path_model)
            index += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Training Sentiment Analysis.")
    parser.add_argument("--embedding_model_path", type=str, required=True,
                        help="path to embedding model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset consist of 'train' and 'val'")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="number of classes to train")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path saved model state_dict")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epoch")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning_rate")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker process data")
    parser.add_argument("--log_iter", type=int, default=8,
                        help="frequence to log.")
    args = parser.parse_args()
    main(args)

# Sentiment Analysis
PhoBERT + 2 layer BiLSTM for Sentiment Analysis 🙆‍♀️
1. Download data from this [link](https://drive.google.com/drive/folders/1FzfKCrA8iVUakvwcQDuUAKF59JzTtNuy?usp=sharing). Extract data and set value of ```SOURCE_FOLDER``` in ```train_lstm_phobert.py``` to path of dataset.
2. Download pretrained model [here](https://drive.google.com/file/d/1FO0CN8kuQ8Jnb1K5YCOp4uundkjUTKJ6/view?usp=sharing). Set value of state_dict_path to the path of pretrained_model.
3. Remember to run ```pip3 install -r requirements.txt``` to install all nescessary packages.

4. To install CocCoc Tokenizer, please run:

```
    git clone https://github.com/coccoc/coccoc-tokenizer.git
```
When coccoc-tokenizer is already cloned. Run these commands on terminal to install it.
```
$ git clone https://github.com/coccoc/coccoc-tokenizer
$ cd coccoc-tokenizer && mkdir build && cd build
$ cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
$ make install
```

5. Now you can train your own model, by running:
```
    python3 train_lstm_phobert.py
```
You can also run to evaluate pretrained model by running:
```
    python3 evaluate.py
```
If you want to test pretrained model on your data. Please put your data into an txt file. You can see an example at ```test.txt```. After that, you can run:
```
    python3 test_lstm_phobert.py
```
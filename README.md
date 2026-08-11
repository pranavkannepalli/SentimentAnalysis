# SentimentAnalysis

Binary text sentiment with a bidirectional LSTM in Keras.

**What it is**
A notebook training a `Bidirectional(LSTM)` classifier on `sentiment_train.csv`. Sentences are tokenized with the Keras `Tokenizer`, mapped to integer sequences, and padded; the model is an `Embedding` (dim 32) into a bi-LSTM (128 units) into a sigmoid output, trained with binary cross-entropy for 5 epochs. It ends by printing a confusion matrix and running a couple of ad-hoc sentence predictions. The exercise was getting a recurrent sequence model working for sentiment.

**Run it**
Open `bidirectional_lstm.ipynb` in Jupyter. Needs `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `seaborn`, `nltk`. Data sits in `data/` (Twitter CSVs are also present). `bert_usage.ipynb` is empty.

As reported in the notebook, the model reaches ~0.98 accuracy on its held-out split, but that split comes from a single small labeled CSV, so treat the number as a training exercise rather than a general benchmark.

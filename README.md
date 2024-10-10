# Airline Reviews Sentiment Analysis
This project aims to perform sentiment analysis on airline customer reviews, using a dataset that contains reviews and recommendations for various airlines. The goal is to classify reviews as positive or negative based on customer recommendations and analyze the sentiment across airlines.

### Data Source
The dataset of airline reviews was collected from https://www.airlinequality.com/review-pages/a-z-airline-reviews/. The processed data is ready for analysis and exploration at https://www.kaggle.com/datasets/juhibhojani/airline-reviews.

## Model
Model Design:
A custom neural network model was designed from scratch using Keras. The architecture includes layers for embedding, LSTM, and Dense layers for binary classification.

Training:
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy

## Benchmarking
By aggregating the Recommended labels, I ranked airlines based on positive recommendations:

## Resources
Environment: Google Colab
Hardware: TPU V2 (RAM: 2.79 GB, Disk: 27.47 GB)
This project used TPUs for efficient training of the model, allowing for the handling of a large dataset with complex deep learning techniques. Further optimization and analysis are suggested to enhance performance and derive more detailed insights from customer reviews.

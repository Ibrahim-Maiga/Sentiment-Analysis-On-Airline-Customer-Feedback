# Airline Reviews Sentiment Analysis

This project focuses on designing and implementing a deep learning model from scratch using Keras and LSTM networks to analyze customer sentiment in the airline industry. The model successfully achieves an accuracy of 86.39% in predicting whether a customer review is positive or negative, based on the likelihood of recommendation. It includes a comprehensive text preprocessing pipeline and features an LSTM-based architecture enhanced with batch normalization and dropout layers for regularization. Hyperparameter tuning is conducted through random search. The dataset comprises 23,171 reviews from 497 airlines, making it well-suited for training and evaluating tools for analyzing customer feedback.

## Data Source
The dataset of airline reviews was collected from https://www.airlinequality.com/review-pages/a-z-airline-reviews/. The processed data is ready for analysis and exploration at https://www.kaggle.com/datasets/juhibhojani/airline-reviews.

## Model

**Model Design:** A custom neural network model was designed from scratch using Keras. The architecture includes layers for embedding, LSTM, and Dense layers for binary classification.

**1. Input Layer:**
- Embedding layer
  - Input dimension: 5000 (vocabulary size)
  - Output dimension: 128 (embedding vector size)
  - Input length: 100 (sequence length)

**2. LSTM Layers:**
- First LSTM layer: 128 units with return_sequences=True
- Second LSTM layer: 64 units

**3. Dense Layers:**
- First dense layer: 32 units with ReLU activation
- Second dense layer: 16 units with ReLU activation
- Output layer: 1 unit with sigmoid activation (for binary classification)

**4. Regularization:**
- Multiple BatchNormalization layers after each major layer
- Dropout layers (0.2 rate) consistently applied throughout

**5. Model Compilation:**
- Optimizer: Adam with learning_rate=0.001
- Loss function: binary_crossentropy
- Metrics: accuracy

The architecture follows a pattern of:
```
Embedding → LSTM → BatchNorm → Dropout → LSTM → BatchNorm → Dropout → 
Dense → BatchNorm → Dropout → Dense → BatchNorm → Dropout → Dense
```

This architecture is structured for sentiment analysis, using:
- Embedding for word representation
- LSTM layers for sequence processing
- Gradually decreasing dense layers (32→16→1)
- Consistent regularization through dropout and batch normalization
- Binary classification output with sigmoid activation

**Benchmarking**

By aggregating the Recommended labels, I ranked airlines based on positive recommendations.

## Key Features

Text preprocessing pipeline for customer review analysis

LSTM-based neural network architecture with batch normalization

Hyperparameter tuning using random search

Model evaluation with comprehensive metrics

Dataset of 23,171 reviews across 497 airlines

**Tech Stack**

* Python
* Keras
* Scikit-learn
* Natural Language Processing

## Resources
Environment: Google Colab

Hardware: TPU V2 (RAM: 2.79 GB, Disk: 27.47 GB)

This project used TPUs for efficient training of the model, allowing for the handling of a large dataset with complex deep learning techniques. Further optimization and analysis are suggested to enhance performance and derive more detailed insights from customer reviews.

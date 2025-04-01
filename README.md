Email Classification Using BERT
This project implements an email classification system using the BERT (Bidirectional Encoder Representations from Transformers) model to classify incoming emails into predefined categories. The model is fine-tuned on a custom dataset containing email text and their respective categories.

Overview
The objective of this project is to build a text classification model that can predict the category of an email based on its content. This is achieved using BERT, a transformer-based model pre-trained on large text corpora, and fine-tuned on a labeled dataset for the specific task.

Approach
Data Preprocessing:

The dataset consists of email texts along with their corresponding categories.

Email text is tokenized using the BERT tokenizer, which converts the text into token IDs that the BERT model can process.

Categories (labels) are encoded into numerical values using LabelEncoder.

Model Selection:

A pre-trained BERT model (bert-base-uncased) is used as the base model for text classification.

The model is fine-tuned on the custom dataset using the CrossEntropy loss function for multi-class classification.

Training:

The dataset is split into training and validation sets.

The model is trained using the AdamW optimizer with a learning rate scheduler to dynamically adjust the learning rate during training.

Evaluation:

The model's performance is evaluated using accuracy, and its predictions are further validated by making predictions on the validation set.

Prediction:

A function is provided to accept user input (email text) and predict its category using the trained model.

Prerequisites
To run this project, you'll need the following libraries:

torch (PyTorch)

transformers (HuggingFace Transformers)

sklearn (Scikit-learn)

pandas

numpy

You can install them using:
pip install torch transformers sklearn pandas numpy

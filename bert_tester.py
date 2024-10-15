import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

# Load BERT model for sentiment analysis (fine-tuned for sentiment tasks)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Pretrained sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
questions_file = "questions.json"
output_file = "responses.json"


# Load json file with questions
def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data


# Dummy response generator (can replace with any response generation logic)
def generate_response(question_text):
    return "This is a sample response for the question: " + question_text


# Sentiment analysis using BERT
def score_kindness(response_text):
    inputs = tokenizer(response_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    # Get the sentiment score (logits)
    sentiment_scores = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(sentiment_scores).item()

    # Mapping sentiment index to label (this model uses 5 labels: 1 star to 5 stars)
    sentiment_labels = {
        0: "very negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "very positive"
    }

    # Calculate kindness score based on sentiment (scaled to 1-10)
    kindness_score = (sentiment + 1) * 2  # Maps 1-star to 2, 5-star to 10, etc.

    return kindness_score, sentiment_labels[sentiment]


# Save questions and responses to file
def save_responses(path_to_json, responses):
    with open(path_to_json, 'w') as f:
        json.dump(responses, f, indent=4)


# Script to process questions and generate responses
def process_questions(input_file, output_file):
    data = load_json(input_file)
    questions = data['questions']
    questions_with_responses = []

    # Iterate over questions and generate responses
    for _ in questions:
        question = _['question']
        response = generate_response(question)
        kindness_score, sentiment_label = score_kindness(response)
        questions_with_responses.append({
            'question': question,
            'response': response,
            'kindness_score': kindness_score,  # Score mapped to a 1-10 scale
            'sentiment': sentiment_label  # Sentiment label (e.g., positive, neutral, etc.)
        })

    save_responses(output_file, questions_with_responses)


if __name__ == "__main__":
    process_questions(questions_file, output_file)
    print(f"Responses saved to {output_file} at {formatted_time}")

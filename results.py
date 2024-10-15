# BASELINE CODE FOR ANY INITIAL RESPONSE TESTING AND SENTIMENT ANALYSIS
# INCLUDES BERT FOR SENTIMENT ANALYSIS (testing)

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datetime import datetime

llama_model_name = "meta-llama/llama-3.2-3b"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

bert_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Pretrained sentiment analysis model
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
questions_file = "questions.json"
output_file = "responses.json"
# load json file with questions
def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data

def generate_response(question_text):
    prompt = f"""
        You are a polite and helpful Medical AI assistant which can book doctor appointments.
        Strictly answer in short. Answer should not exceed 3 sentences.
        Explain medical terms to user if user asks for explanation.
        To book appointment you need date, time, doctor's name, doctor's domain and insurance number from user.
        If date, time, doctor's name, doctor's domain and insurance number is present in the user request, generate the
        final appointment details.
        If this information is not present, ask user question to get this information.
        Do not hallucinate the appointment details. These details should come from user only.
        Convert vague appointment days to exact date. Example if user says tomorrow, then use today's date
        and calculate tomorrow's date.
        Today's date in format of month-date-year.
        If the doctor's name or domain is not given, then based on symptoms suggest a domain expert doctor.
        Following are the available domain experts and their available times:  
        [Jacob, cardiologist, Friday 1pm to 3pm], [John, neurologist, Thursday 9am to 2pm], 
        [Jennifer, pulmonologist, Wednesday 10am to 11am], [House, Neurosurgeon, Tuesday 9am to 5pm], 
        [Olivia, orthopedic surgeon, Monday 1 pm to 4pm], [Alisha, dermatologist, Friday 9 am to 1 pm], 
        [Mark, gastroenterologist, Tuesday 8 am to 3 pm], [Ryan, general doctor, Monday 10 am to 5 pm].
        The appointment duration is 30 minutes.
        So the appointment booking should be within these times; it cannot go beyond this.
        For example, appointment for Jacob is booked at 3pm, it is not valid,
        since the doctor has to attend the patient from 3pm to later.
        Book such that the appointment ends within the available time.
        Strictly do not suggest any other doctor or domain of the doctor other than the symptoms and the provided list.
        Strictly be concise in question. Use questions with few words.
        Strictly answer in short and ask questions in short.
        Confirm the final details with the user and generate a json response.
        The json response should have date, time, doctor's name, doctor's domain, patient’s name, patient’s phone number or email, and insurance number as keys.

        User Question: "{question_text}"
        """

    inputs = llama_tokenizer(prompt, return_tensors="pt")
    output = llama_model.generate(**inputs, max_length=100, temperature=0.9, num_return_sequences=1)

    # decode response
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# score kindness using LLM given response text
def score_kindness_llm(response_text):
    prompt = f"""
        You are a highly empathetic AI trained to assess the kindness in responses.
        Rate the following response on a scale of 1 to 10 where 1 is not kind at all and 10 is extremely kind. 
        Please consider tone, politeness, and empathy.

        Response: "{response_text}"

        Kindness Score (1-10):"""
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    output = llama_model.generate(**inputs, max_length=100, temperature=0.9, num_return_sequences=1)

    # decode score
    kindness_score = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return kindness_score

# Sentiment analysis using BERT
def score_kindness_bert(response_text):
    inputs = bert_tokenizer(response_text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)

    # Get the sentiment score (logits)
    sentiment_scores = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(sentiment_scores).item()

    # Mapping sentiment index to label (1 to 5 stars
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

# save questions and responses to file
def save_responses(path_to_json, responses):
    with open(path_to_json, 'w') as f:
        json.dump(responses, f, indent=4)

# script to process questions and generate responses:
def process_questions_llm(input_file, output_file):
    data = load_json(input_file)
    questions = data['questions']
    questions_with_responses = []

    # iterate over questions and generate responses
    for _ in questions:
        question = _['question']
        response = generate_response(question)
        kindness_score = score_kindness_llm(response)
        questions_with_responses.append({
            'question': question,
            'response': response,
            'kindness_score': kindness_score.strip()
        })

    save_responses(output_file, questions_with_responses)

def process_questions_bert(input_file, output_file):
    data = load_json(input_file)
    questions = data['questions']
    questions_with_responses = []

    # iterate over questions and generate responses
    for _ in questions:
        question = _['question']
        response = generate_response(question)
        kindness_score, sentiment_label = score_kindness_bert(response)
        questions_with_responses.append({
            'question': question,
            'response': response,
            'kindness_score': kindness_score,
            'sentiment': sentiment_label
        })

    save_responses(output_file, questions_with_responses)

if __name__ == "__main__":
    process_questions_llm(questions_file, output_file)
    # process_questions.bert(questions_file, output_file)
    print(f"Responses saved to {output_file} at {formatted_time}")

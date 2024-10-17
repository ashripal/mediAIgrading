# MULTI-AGENT SYSTEM FOR APPOINTMENT BOOKING AND DIAGNOSIS TASKS
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertForSequenceClassification, BertTokenizer
from datetime import datetime

# load LLAMA-3.2-3b for doctor role
doctor_model = "meta-llama/llama-3.2-3b"  # Pretrained model for doctor role
doctor_tokenizer = AutoTokenizer.from_pretrained(doctor_model)
doctor_model = AutoModelForCausalLM.from_pretrained(doctor_model)

# load small LLM for patient role (simulate simple responses)
patient_model = "meta-llama/llama-2-7b"
patient_tokenizer = AutoTokenizer.from_pretrained(patient_model)
patient_model = AutoModelForCausalLM.from_pretrained(patient_model)

# load BERT for sentiment analysis
bert_model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
bert_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
questions_file = "questions.json"
output_file = "conversations.json"

def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data

llm_prompts = load_json("LLMPrompts.json")

# generate patient responses given history and symptoms
# def generate_patient_response(history, symptoms):
#     input_text = f"Patient history: {history} Symptoms: {symptoms}"
#     inputs = patient_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
#     outputs = patient_model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True)
#     response = patient_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# generate a patient response based on the type of question
def generate_patient_response(question_text, response_type):
    # Define different speech styles in prompts
    if response_type == 'simple':
        style_prompt = "The patient is responding in clear, simple English. No complex words."
    elif response_type == 'lingo':
        style_prompt = "The patient uses slang in their response. Use informal language and casual tone."
    elif response_type == 'filler':
        style_prompt = "The patient is hesitant and uses filler words like 'um', 'like', and 'uh'."

    prompt = f"""
    You are a patient having a conversation with a doctor. The doctor has asked the following question:
    {question_text}

    {style_prompt}
    Respond to the question in 1 to 2 sentences in the given style.
    """
    inputs = patient_tokenizer(prompt, return_tensors="pt")
    output = patient_model.generate(**inputs, max_length=100, temperature=0.9, num_return_sequences=1)

    # decode response
    response = patient_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# generate doctor responses given patient responses
def generate_doctor_response(patient_response):
    doctor_prompt = f"You are a doctor talking to a patient who is looking to book an appointment. The patient has the following response upon calling you: {patient_response}. \nWhat is your next question?"
    inputs = doctor_tokenizer(doctor_prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = doctor_model.generate(**inputs, max_length=150)
    response = doctor_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Sentiment analysis using BERT
def score_kindness(response_text):
    inputs = bert_tokenizer(response_text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)

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

def save_conversations(path_to_json, conversations):
    with open(path_to_json, 'w') as f:
        json.dump(conversations, f, indent=4)

# process multi-turn conversation between patient and doctor
# def simulate_conversation(history, symptoms):
#     patient_history = history
#     patient_symptoms = symptoms
#     patient_response = generate_patient_response(history, symptoms)
#     conversation_log = []
#     for turn in range(5):   # simulate 5 turns of conversation
#         doctor_response = generate_doctor_response(patient_response)
#         kindness_score, sentiment_label = score_kindness(doctor_response)
#         conversation_log.append({
#             "patient_response": patient_response,
#             "doctor_response": doctor_response,
#             "kindness_score": kindness_score,
#             "sentiment_label": sentiment_label
#         })
#         patient_response = generate_patient_response(patient_history, patient_symptoms)
#
#     return conversation_log

# MODIFIED FOR SIMPLE, LINGO, AND FILLER RESPONSES
def process_conversations(input_file, output_file):
    data = load_json(input_file)
    questions = data['questions']
    conversations = []

    # iterate over questions and generate patient responses
    for _ in questions:
        question = _['question']
        response_type = _['type']  # simple, lingo, filler
        response = generate_patient_response(question, response_type)

        # Store the entire conversation
        conversations.append({
            'question': question,
            'response_type': response_type,
            'patient_response': response
        })

    save_conversations(output_file, conversations)

if __name__ == "__main__":
    process_conversations(questions_file, output_file)
    print(f"Conversations saved to {output_file} at {formatted_time}")
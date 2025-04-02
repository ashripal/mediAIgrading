import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Required fields for appointment JSON validation
REQUIRED_FIELDS = ["date", "time", "doctor", "domain", "patient_name", "contact", "insurance_number"]

def llama_generate(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Define the function to evaluate kindness
def evaluate_conversation_kindness(conversation):
    """
    Evaluates the kindness of a conversation using GPT-4 and returns a score.
    The score ranges from 0 (not kind) to 10 (extremely kind).
    """
    prompt = f"""
You are a helpful and impartial assistant. Please evaluate the following conversation for its kindness.

Kindness is defined as:
- Showing empathy and understanding.
- Using polite and respectful language.
- Offering help or support in a friendly manner.

Conversation:
{conversation}

Score the kindness of this conversation on a scale from 0 to 10, where:
- 0 means the conversation lacks kindness completely.
- 10 means the conversation is extremely kind and empathetic.

Explain your reasoning for the score.

Your output should be in the following JSON format:
{{
    "kindness_score": <score>,
    "reasoning": "<your reasoning>"
}}
"""
    model_output = llama_generate(prompt, max_length=500)
    try:
        evaluation = json.loads(model_output)  # Parse the JSON response
        return evaluation
    except Exception as e:
        print(f"Error parsing kindness response: {e}")
        return None

# Validate if the JSON is complete
def validate_json_completeness(appointment_json):
    """
    Checks if all required fields exist in the JSON.
    """
    if not appointment_json:
        return {"json_generated": False, "missing_fields": REQUIRED_FIELDS, "completeness_score": 0}

    missing_fields = [field for field in REQUIRED_FIELDS if field not in appointment_json]
    completeness_score = 10 if not missing_fields else (10 - (len(missing_fields) * 2))

    return {
        "json_generated": True,
        "missing_fields": missing_fields,
        "completeness_score": max(completeness_score, 0),
    }


# Check if the conversation is too long (conciseness)
def evaluate_conciseness(conversation):
    """
    Determines if the AI assistant's responses are too verbose.
    """
    word_count = len(conversation.split())
    max_words = 300  # Define a reasonable conversation length limit

    conciseness_score = max(0, 10 - (word_count - max_words) // 50) if word_count > max_words else 10
    return {"word_count": word_count, "conciseness_score": conciseness_score}


# Check if doctor specialization matches symptoms
def evaluate_accuracy(appointment_json, conversation):
    """
    Evaluates whether the doctor's specialization aligns with the symptoms mentioned.
    """
    doctor_specializations = {
        "Jacob": "Cardiologist",
        "John": "Neurologist",
        "Jennifer": "Pulmonologist",
        "House": "Neurosurgeon",
        "Olivia": "Orthopedic Surgeon",
        "Alisha": "Dermatologist",
        "Mark": "Gastroenterologist",
        "Ryan": "General Doctor",
    }

    symptoms_to_specialty = {
        "chest pain": "Cardiologist",
        "shortness of breath": "Pulmonologist",
        "headache": "Neurologist",
        "back pain": "Orthopedic Surgeon",
        "skin rash": "Dermatologist",
        "stomach pain": "Gastroenterologist",
    }

    # Extract assigned doctor and symptoms from conversation
    doctor_name = appointment_json.get("doctor", "").strip()
    assigned_specialty = appointment_json.get("domain", "").strip()

    # Check if symptoms mentioned in conversation match the doctor's specialization
    matching_specialty = None
    for symptom, specialty in symptoms_to_specialty.items():
        if symptom in conversation.lower():
            matching_specialty = specialty
            break

    accuracy_score = 10 if doctor_specializations.get(doctor_name) == assigned_specialty and assigned_specialty == matching_specialty else 5
    return {"assigned_specialty": assigned_specialty, "matching_specialty": matching_specialty, "accuracy_score": accuracy_score}


# alternate test for doctor selection
def evaluate_doctor_selection_gpt(conversation):
    """
    Uses GPT-4 to verify if the AI assistant assigned the correct doctor based on symptoms.
    """
    prompt = f"""
You are a medical AI evaluator. Analyze the following conversation to determine if the AI assistant correctly assigned a doctor based on the patient's symptoms.

Conversation:
{conversation}

Evaluate if the selected doctor matches the symptoms correctly.
Provide a score (0-10) and reasoning in this JSON format:
{{
    "doctor_selection_score": <score>,
    "reasoning": "<your reasoning>"
}}
"""

    model_output = llama_generate(prompt, max_length=500)
    try:
        evaluation = json.loads(model_output)  # Parse the JSON response
    except Exception as e:
        print(f"Error parsing doctor selection response: {e}")
        return {"doctor_selection_score": 0, "reasoning": "Evaluation failed."}

# Main function to evaluate conversation
def evaluate_conversation(conversation):
    """
    Evaluates the conversation across multiple categories and returns a consolidated report.
    """
    evaluation_report = {}

    # Kindness evaluation
    kindness_eval = evaluate_conversation_kindness(conversation)
    evaluation_report.update(kindness_eval if kindness_eval else {"kindness_score": 0, "reasoning": "Evaluation failed."})

    # Attempt to extract JSON from conversation
    try:
        json_start = conversation.find("{")
        json_end = conversation.rfind("}") + 1
        appointment_json = json.loads(conversation[json_start:json_end]) if json_start != -1 and json_end > json_start else None
    except Exception:
        appointment_json = None

    # JSON completeness evaluation
    completeness_eval = validate_json_completeness(appointment_json)
    evaluation_report.update(completeness_eval)

    # Conciseness evaluation
    conciseness_eval = evaluate_conciseness(conversation)
    evaluation_report.update(conciseness_eval)

    # Accuracy evaluation
    accuracy_eval = evaluate_accuracy(appointment_json, conversation)
    evaluation_report.update(accuracy_eval)

    evaluation_report.update(evaluate_doctor_selection_gpt(conversation))

    # Calculate overall score (average of all categories)
    scores = [evaluation_report.get("kindness_score", 0),
              evaluation_report.get("completeness_score", 0),
              evaluation_report.get("conciseness_score", 0),
              evaluation_report.get("accuracy_score", 0),
              evaluation_report.get("doctor_selection_score", 0)]
    evaluation_report["overall_score"] = round(sum(scores) / len(scores), 2)

    return evaluation_report

if __name__ == "__main__":
    # Example conversation
    conversation = """
    AI Assistant: Hello! How can I assist you today?
    Patient: Hi, I’m feeling very unwell. I have chest pain and difficulty breathing.
    AI Assistant: I’m sorry to hear that. I’ll do my best to assist you. Let’s find the right doctor for you.
    Patient: Thank you, I really appreciate your help.
    AI Assistant: You’re welcome. It’s my pleasure to help. Based on your symptoms, I recommend seeing a pulmonologist. Would Wednesday at 10 AM with Dr. Jennifer work for you?
    Patient: Yes, that works. Thank you again.
    AI Assistant: Perfect. I’ll book your appointment and send you the details shortly. Take care and feel better soon!

    {
        "date": "12/03/2024",
        "time": "10:00 AM",
        "doctor": "Jennifer",
        "domain": "Pulmonologist",
        "patient_name": "John Doe",
        "contact": "johndoe@example.com",
        "insurance_number": "12345"
    }
    """

    # Evaluate the conversation
    evaluation = evaluate_conversation(conversation)

    # Output the result
    if evaluation:
        print("\n📊 **Evaluation Report:**")
        print(json.dumps(evaluation, indent=4))
    else:
        print("Failed to evaluate the conversation.")

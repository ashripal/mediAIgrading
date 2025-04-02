import os
import json
import evaluator_llama
from evaluator_llama import evaluate_conversation_kindness, validate_json_completeness, evaluate_conciseness, evaluate_accuracy
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Base Prompt Definition
base_prompt = """
You are an extremely polite, emphathetic and helpful Medical AI assistant tasked with handling patient queries to book doctor appointments. Follow these strict instructions:

Input and Output: Generate a simulated dialogue between the AI assistant and the patient. The conversation must start naturally, with the AI assistant greeting the patient and the patient explaining their symptoms or request.

Your job is to extract all required details to finalize the appointment:
- Date
- Time
- Doctor's name
- Doctor's domain (specialization)
- Patient’s name
- Contact (email or phone number)
- Insurance number

Doctor Availability:
- Jacob (Cardiologist): Friday 1pm - 3pm
- John (Neurologist): Thursday 9am - 2pm
- Jennifer (Pulmonologist): Wednesday 10am - 11am
- House (Neurosurgeon): Tuesday 9am - 5pm
- Olivia (Orthopedic Surgeon): Monday 1pm - 4pm
- Alisha (Dermatologist): Friday 9am - 1pm
- Mark (Gastroenterologist): Tuesday 8am - 3pm
- Ryan (General Doctor): Monday 10am - 5pm

Symptoms and Doctor Matching:
Based on the doctor’s specialization, generate plausible symptoms the patient might be experiencing.

Conversation Flow:
The conversation must begin with the AI assistant greeting the patient and asking how it can assist. Use natural language and ensure all required details are collected. Confirm the final appointment details in the following JSON structure:

{
    "date": "DD/MM/YYYY",
    "time": "HH:MM AM/PM",
    "doctor": "Doctor Name",
    "domain": "Doctor's Specialization",
    "patient_name": "Patient Name",
    "contact": "Email or Phone",
    "insurance_number": "Insurance Number"
}

If the necessary information is not present, ask the user questions to fill in the missing details. End the conversation by confirming the appointment details. Ensure that the date and time are within the doctor's availability.
The appointment duration is 30 minutes and cannot go overtime. Book appointments that end within the doctor's available time slots. Be consise while asking for the user's details.

These dates are for a past year, 2024. Ensure that the dates are in correct format and that they are on the specified day of the appointment. The time should be in AM/PM format.

Example Conversation:
Agent: Hello! How can I assist you today?
Patient: Hi, I’m having chest pain and feeling shortness of breath. I think I need to see a doctor.

Agent: I’m sorry to hear that. Based on your symptoms, a cardiologist would be best to consult. Dr. Jacob, our cardiologist, is available on Fridays from 1 PM to 3 PM. Would that work for you?
Patient: Yes, Friday at 1 PM sounds good.

Agent: Perfect. Could you please provide your full name for the appointment?
Patient: Sure, it’s Sarah Connor.

Agent: Thank you, Sarah. May I have your contact details, such as your phone number or email address?
Patient: You can reach me at 555-123-4567.

Agent: Got it. Lastly, do you have an insurance number you’d like to use for this appointment?
Patient: Yes, my insurance number is INS-987654321.

Agent: Thank you for sharing the details. Let me confirm your appointment:

Date: This Friday
Time: 1:00 PM
Doctor: Dr. Jacob
Specialization: Cardiologist
Patient Name: Sarah Connor
Contact: 555-123-4567
Insurance Number: INS-987654321
Does everything look correct?
Patient: Yes, that’s perfect.

Agent: Great! Your appointment is confirmed. Please arrive 10 minutes early and bring your insurance card. Have a great day, Sarah!
"""

# Function to generate conversation using GPT-4
def generate_conversation(prompt, max_tokens=1000):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_tokens, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Save JSON to file
def save_json_to_file(data, filename="appointment.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"📂 Appointment details saved to {filename}")

# Improve the prompt dynamically based on evaluation results
def improve_prompt(conversation, prompt, evaluation):
    improvements = {
        "kindness_score": {
            "low_threshold": 7,
            "suggestions": "Use more empathetic and friendly language, like 'I understand' or 'I'm here to help.'"
        },
        "completeness_score": {
            "low_threshold": 9,
            "suggestions": "Ensure that all necessary appointment details (date, time, doctor, domain, patient name, contact, insurance) are extracted and confirmed explicitly."
        },
        "conciseness_score": {
            "low_threshold": 7,
            "suggestions": "Avoid unnecessary verbosity. Keep responses direct while still being polite."
        },
        "accuracy_score": {
            "low_threshold": 9,
            "suggestions": "Verify that the assigned doctor’s specialization matches the patient’s symptoms. Correct any mismatches."
        },

        "doctor_selection_score": {
            "low_threshold": 9,
            "suggestions": "Double-check the symptoms and ensure that the recommended doctor’s specialization matches them appropriately. If uncertain, ask the patient for more details before assigning a doctor."
        }
    }

    new_prompt = prompt

    # Apply modifications to the prompt based on low scores
    for category, details in improvements.items():
        if category in evaluation and evaluation[category] < details["low_threshold"]:
            print(f"Improving {category}: Current score = {evaluation[category]}")
            new_prompt += f"\n{details['suggestions']}"

    return new_prompt

# RL loop: iteratively improve the conversation until a high score is achieved
max_iterations = 5
iteration = 0
current_prompt = base_prompt
high_score_threshold = 8  # Minimum acceptable score for all categories

while iteration < max_iterations:
    print(f"\nIteration {iteration + 1}: Generating conversation...")
    conversation_output = generate_conversation(current_prompt)

    # Evaluate the conversation
    evaluation = evaluator.evaluate_conversation(conversation_output)

    # Extract JSON from the conversation
    try:
        json_start = conversation_output.find("{")
        json_end = conversation_output.rfind("}") + 1
        appointment_json = json.loads(conversation_output[json_start:json_end]) if json_start != -1 and json_end > json_start else None
    except Exception:
        appointment_json = None

    if not appointment_json:
        print("\nJSON was not generated or is malformed. Improving prompt and retrying...")
        current_prompt += "\nEnsure that the conversation includes a structured JSON output at the end."
        iteration += 1
        continue

    # Save and validate extracted appointment JSON
    save_json_to_file(appointment_json)

    # Check if all evaluation scores meet the threshold
    if all(evaluation.get(category, 0) >= high_score_threshold for category in evaluation):
        print("\nHigh scores achieved! Final optimized prompt is ready.")
        break
    else:
        print("\nImproving prompt based on evaluation feedback...")
        current_prompt = improve_prompt(conversation_output, current_prompt, evaluation)

    iteration += 1

if iteration == max_iterations:
    print("\nMaximum iterations reached. The prompt is not fully optimized but has been improved.")

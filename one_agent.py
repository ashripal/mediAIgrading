from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from huggingface_hub import login


#login(token=token)

# Load the Llama model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=token)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=token)

# Define the prompt
prompt = """
You are a Medical AI assistant tasked with handling patient queries to book doctor appointments. Follow these strict instructions:

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

Example Start:
AI Assistant: "Hello! How can I assist you today?"
Patient: "Hi, I’m having chest pain and shortness of breath."
"""

# Generate conversation using the model
def generate_conversation(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Get the response from the model
conversation_output = generate_conversation(prompt)

# Print the output
print("Generated Conversation:")
print(conversation_output)

# Extract JSON response from the conversation (if included)
try:
    json_start = conversation_output.find("{")
    json_end = conversation_output.rfind("}") + 1
    appointment_json = json.loads(conversation_output[json_start:json_end])
    print("\nExtracted Appointment Details:")
    print(json.dumps(appointment_json, indent=4))
except Exception as e:
    print("\nFailed to extract JSON response. Check the conversation output.")

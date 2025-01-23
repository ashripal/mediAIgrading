import openai
import json

# Set your OpenAI API key
#openai.api_key = "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"

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

# Generate conversation using GPT-4
def generate_conversation(prompt, max_tokens=1000):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

# Validate JSON response for all necessary fields
def validate_appointment_json(appointment_json):
    required_fields = [
        "date", "time", "doctor", "domain",
        "patient_name", "contact", "insurance_number"
    ]
    missing_fields = [field for field in required_fields if field not in appointment_json]
    if missing_fields:
        raise ValueError(f"Missing fields in the appointment JSON: {missing_fields}")
    print("Validation successful: All required fields are present.")

# Save JSON to file
def save_json_to_file(data, filename="appointment.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Appointment details saved to {filename}")

# Get the response from GPT-4
conversation_output = generate_conversation(prompt)

# Print the output
print("Generated Conversation:")
print(conversation_output)

# Extract JSON response from the conversation
try:
    json_start = conversation_output.find("{")
    json_end = conversation_output.rfind("}") + 1
    appointment_json = json.loads(conversation_output[json_start:json_end])
    print("\nExtracted Appointment Details:")
    print(json.dumps(appointment_json, indent=4))

    # Validate the JSON and save to file
    validate_appointment_json(appointment_json)
    save_json_to_file(appointment_json)
except Exception as e:
    print(f"\nError: {e}\nFailed to extract or validate JSON response. Check the conversation output.")

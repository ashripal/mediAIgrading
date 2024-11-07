import json
from datetime import datetime

# Sample test data with multiple entries
test_data_list = [
    {
        "question": "Hi, I'd like to book an appointment with a general physician. I'm free on November 3rd in the morning.",
        "symptoms": "Mild headache and fatigue",
        "expected_output": {
            "date": "November 3",
            "time": "10:00 AM",
            "doctor": "Ryan",
            "domain": "general doctor",
            "patient_name": "Alice Smith",
            "contact": "alice@example.com",
            "insurance_number": "INS123456"
        }
    },
    {
        "question": "Hello, can I schedule a visit with a cardiologist next Friday afternoon?",
        "symptoms": "Chest pain and shortness of breath",
        "expected_output": {
            "date": "November 5",
            "time": "1:00 PM",
            "doctor": "Jacob",
            "domain": "cardiologist",
            "patient_name": "Bob Johnson",
            "contact": "bob@example.com",
            "insurance_number": "INS654321"
        }
    },
    {
        "question": "I need to see a dermatologist. Are there any slots available tomorrow?",
        "symptoms": "Skin rash and itching",
        "expected_output": {
            "date": "November 2",
            "time": "9:00 AM",
            "doctor": "Alisha",
            "domain": "dermatologist",
            "patient_name": "Carol White",
            "contact": "carol@example.com",
            "insurance_number": "INS987654"
        }
    }
]

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output_file = "conversations.json"

# Doctor's available slots
available_doctors = {
    "Jacob": {"domain": "cardiologist", "availability": ["Friday 1pm to 3pm"]},
    "John": {"domain": "neurologist", "availability": ["Thursday 9am to 2pm"]},
    "Jennifer": {"domain": "pulmonologist", "availability": ["Wednesday 10am to 11am"]},
    "House": {"domain": "neurosurgeon", "availability": ["Tuesday 9am to 5pm"]},
    "Olivia": {"domain": "orthopedic surgeon", "availability": ["Monday 1pm to 4pm"]},
    "Alisha": {"domain": "dermatologist", "availability": ["Friday 9am to 1pm"]},
    "Mark": {"domain": "gastroenterologist", "availability": ["Tuesday 8am to 3pm"]},
    "Ryan": {"domain": "general doctor", "availability": ["Monday 10am to 5pm"]},
}

def generate_doctor_response(question_text):
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
[Jennifer, pulmonologist, Wednesday 10am to 11am], [House, neurosurgeon, Tuesday 9am to 5pm],
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

    # For the purpose of this example, we will simulate the doctor's response.
    # In a real application, this is where you'd integrate the LLM to generate the response.
    # Since we are to generate the necessary output immediately, we'll assume the patient provides all required information.

    # Extract necessary details from the question (this is simplified and assumes perfect input)
    # In reality, you'd need NLP to parse dates, times, etc.

    # Mocked data based on the question
    if "general physician" in question_text or "general doctor" in question_text:
        doctor_name = "Ryan"
        domain = "general doctor"
    elif "cardiologist" in question_text:
        doctor_name = "Jacob"
        domain = "cardiologist"
    elif "dermatologist" in question_text:
        doctor_name = "Alisha"
        domain = "dermatologist"
    else:
        doctor_name = "Unknown"
        domain = "general doctor"

    # Mocked appointment date and time (would normally parse from question)
    if "tomorrow" in question_text:
        appointment_date = (datetime.now() + timedelta(days=1)).strftime("%B %d")
    else:
        appointment_date = "November 3"
    appointment_time = "10:00 AM"

    # Collect patient's name, contact, and insurance number (assuming provided)
    patient_name = "John Doe"
    contact = "john.doe@example.com"
    insurance_number = "INS000000"

    # Generate the JSON response
    response_json = {
        "date": appointment_date,
        "time": appointment_time,
        "doctor": doctor_name,
        "domain": domain,
        "patient_name": patient_name,
        "contact": contact,
        "insurance_number": insurance_number
    }

    # Confirm the final details with the user (simulated)
    confirmation_message = f"Your appointment with Dr. {doctor_name}, a {domain}, is booked for {appointment_date} at {appointment_time}. Please confirm your name, contact, and insurance number."

    response = {
        "prompt": prompt,
        "doctor_response": confirmation_message,
        "appointment_details": response_json
    }
    return response

# Function to evaluate the generated response against the expected output
def evaluate_responses(generated_output, expected_output):
    # Compare the appointment details
    match = generated_output["appointment_details"] == expected_output
    return 1 if match else 0

# Process multiple conversations and store them
def process_conversations(test_data_list):
    conversations = []

    for test_data in test_data_list:
        question = test_data["question"]
        expected_output = test_data["expected_output"]

        # Generate doctor response
        doctor_response = generate_doctor_response(question)

        # Evaluation of response against the expected output
        match_score = evaluate_responses(doctor_response, expected_output)

        # Store the conversation and results
        conversation = {
            "question": question,
            "doctor_response": doctor_response["doctor_response"],
            "appointment_details": doctor_response["appointment_details"],
            "expected_output": expected_output,
            "match_score": match_score
        }
        conversations.append(conversation)

    # Save conversations to JSON file
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=4)
    print(f"Conversations saved to {output_file} at {formatted_time}")

if __name__ == "__main__":
    process_conversations(test_data_list)

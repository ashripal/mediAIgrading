import openai
import json

# SET UP AND USE OPENAI API
#openai.api_key = ""

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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the model's response
    try:
        model_output = response['choices'][0]['message']['content']
        evaluation = json.loads(model_output)  # Parse the JSON response
        return evaluation
    except Exception as e:
        print(f"Error parsing the response: {e}")
        return None

# Example conversation
conversation = """
AI Assistant: Hello! How can I assist you today?
Patient: Hi, I’m feeling very unwell. I have chest pain and difficulty breathing.
AI Assistant: I’m sorry to hear that. I’ll do my best to assist you. Let’s find the right doctor for you.
Patient: Thank you, I really appreciate your help.
AI Assistant: You’re welcome. It’s my pleasure to help. Based on your symptoms, I recommend seeing a cardiologist. Would Friday at 1 PM with Dr. Jacob work for you?
Patient: Yes, that works. Thank you again.
AI Assistant: Perfect. I’ll book your appointment and send you the details shortly. Take care and feel better soon!
"""

# Evaluate the kindness of the conversation
evaluation = evaluate_conversation_kindness(conversation)

# Output the result
if evaluation:
    print("\nKindness Evaluation:")
    print(json.dumps(evaluation, indent=4))
else:
    print("Failed to evaluate the conversation.")

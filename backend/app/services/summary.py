import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_meeting_summary(transcript_text: str) -> dict:
    if not transcript_text.strip():
        return {"summary": "No transcript available.", "action_items": [], "key_decisions": []}
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""You are an expert AI Meeting Assistant. Read the following meeting transcript and extract:
1. A brief summary (2-3 sentences)
2. A list of action items (assigned to specific people if possible)
3. Key decisions made

Format your output strictly as a JSON object with these exact keys:
{{
    "summary": "string",
    "action_items": ["item 1", "item 2"],
    "key_decisions": ["decision 1", "decision 2"]
}}

Transcript:
{transcript_text}
"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
            temperature=0.3
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return {"summary": "Failed to generate summary.", "action_items": [], "key_decisions": []}


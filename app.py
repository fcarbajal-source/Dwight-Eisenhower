import streamlit as st
from google import genai
from google.genai import types

# -------------------------------------------------------------
# Configuration & Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Ask Ike: Dwight D. Eisenhower",
    page_icon="⭐",
    layout="centered"
)

# Load API key safely from Streamlit secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY. Please configure it in .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Configured Model ID (per instructions)
MODEL_ID = "gemini-3.5-flash-lite"

# -------------------------------------------------------------
# System Prompt (Persona, Grade Level, Archives, Constraints)
# -------------------------------------------------------------
SYSTEM_PROMPT = """
You are Dwight D. Eisenhower ("Ike") — Five-Star General, Supreme Allied Commander during WWII, and 34th President of the United States.

Target Audience & Voice:
- Speak directly to 10th-grade high school students in the United States.
- Match Eisenhower's authentic speaking style: measured, plain-spoken, principled, warm, and resolute, but explain complex military, foreign policy, and constitutional concepts using clear vocabulary suitable for a 15-to-16-year-old reader.
- Use first-person ("I", "my administration", "when I commanded Allied forces").

Historical Grounding & Knowledge Sources:
Base all facts and answers strictly on reputable primary and secondary sources, specifically:
- Eisenhower Presidential Library Online Documents & Papers
- The American Presidency Project (UC Santa Barbara)
- Library of Congress Eisenhower Resource Guide
- National Archives and Records Administration (NARA) & DocsTeach Eisenhower Collection
- Foreign Relations of the United States (FRUS) Eisenhower Administration Volumes
- Eisenhower Foundation Primary Source Library
- The Papers of Dwight David Eisenhower (Johns Hopkins University Press)
- Teaching American History Eisenhower Archive
- Columbia University Eisenhower Administration Oral History Project
- His official letters, speeches (e.g., D-Day Order of the Day, Atoms for Peace, Farewell Address), legislation signed (e.g., Federal-Aid Highway Act of 1956, Civil Rights Acts of 1957 and 1960), and Executive Orders (e.g., EO 10730 regarding Little Rock Central High School).

Strict Output Rules:
1. Length: Every answer MUST be between 15 and 120 words total.
2. Citation: Always conclude the answer with a source attribution on a new line in brackets, indicating the historical basis (e.g., "[Source: Farewell Address, January 17, 1961]" or "[Source: Letter to Hazlett, Eisenhower Presidential Library]"). The citation counts toward the word count.
3. Stay strictly in character. Never mention that you are an AI or LLM.
"""

# -------------------------------------------------------------
# App Header
# -------------------------------------------------------------
st.title("⭐ Dwight D. Eisenhower: High School History Chat")
st.caption("Talk to President Eisenhower about WWII, the Cold War, Civil Rights, and leadership.")

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
# st.session_state.messages will store dicts: {"role": "user"|"assistant", "content": str}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------------------------------------
# Chat Input & Logic
# -------------------------------------------------------------
if prompt := st.chat_input("Ask General Eisenhower a question..."):
    # Render user query
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Memory limit: Only keep the previous 6 questions and their corresponding replies
    # (6 user queries + up to 6 assistant answers = maximum 12 messages sent to API)
    trimmed_history = st.session_state.messages[-12:]

    # Prepare chat contents in the format expected by Google GenAI SDK
    formatted_contents = []
    for msg in trimmed_history:
        role = "user" if msg["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Request completion
    with st.chat_message("assistant"):
        with st.spinner("General Eisenhower is considering his reply..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                    ),
                )
                reply = response.text
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")

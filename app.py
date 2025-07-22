import streamlit as st
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict
import re
import os

# Page configuration
st.set_page_config(
    page_title="The Chronicler of the Nile",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and Prompts
CHRONICLER_SYSTEM_PROMPT = """You are "The Chronicler of the Nile" — a sophisticated digital historian with complete mastery over Egyptian history across all its eras, from the Pharaonic and Graeco-Roman periods through the Islamic, Ottoman, and modern eras.

Your core responsibilities and behavior are as follows:

— KNOWLEDGE SCOPE —
You possess accurate, in-depth knowledge of all major periods of Egyptian history:
1. **Ancient Egypt** – Pharaohs, dynasties, gods, pyramids, daily life, religious practices.
2. **Graeco-Roman Egypt** – The Ptolemaic dynasty, Cleopatra, Roman conquest, early Christianity.
3. **Islamic Egypt** – The Arab conquest, Fatimid, Ayyubid, Mamluk, and Ottoman periods, key figures, Islamic culture, architecture, and society.
4. **Modern Egypt** – The Muhammad Ali dynasty, British occupation, 1919 and 1952 revolutions, Nasser, Sadat, Mubarak, and events up to the early 21st century (cutoff at 2011).

— PERSONALITY & TONE —
You are knowledgeable, authoritative, and objective. You do not use modern slang, but you are not archaic or dramatic. For ancient topics, you may evoke a sense of grandeur, but you must shift to a clear, academic tone for later eras. You are always respectful, factual, and neutral in tone.

— BEHAVIORAL RULES —
1. **Always provide factual, historically accurate answers.**
   - When historical debate exists, present multiple scholarly views.
   - Example: "Most historians believe X, though others argue Y."

2. **Never comment on current political events or give personal opinions.**
   - Never make predictions or speculate on future events.
   - Only offer "what-if" analysis if explicitly prompted, and clearly state it is hypothetical.

3. **Handle unclear or out-of-scope queries gracefully.**
   - If the question is ambiguous, ask for clarification.
   - If a topic is beyond your scope, say: "My records do not contain sufficient detail on that subject."

4. **Strict Historical Cutoff (2011):**
   - For any questions about events after 2011, respond: "My historical records conclude in 2011. I cannot provide information about more recent events."
   - This includes questions about current presidents, recent political developments, or contemporary issues.

5. **STRICT TOPIC FILTERING - MOST IMPORTANT:**
   You ONLY answer questions about Egyptian history. For any question unrelated to Egypt or Egyptian history, respond with:
   "I am The Chronicler of the Nile, specializing exclusively in Egyptian history across all eras. I cannot assist with topics outside Egyptian history. Please ask me about pharaohs, dynasties, monuments, Islamic Egypt, modern Egyptian history, or any aspect of Egypt's rich historical heritage."

— CONVERSATIONAL MEMORY —
You must retain the full context of the conversation (both user and your own replies). If the session becomes too long, you should:
- Summarize earlier parts of the conversation to preserve meaning.
- Retain important context from the beginning (e.g., chosen time period, figures mentioned).
This allows you to reference past turns and build understanding across multi-turn dialogues.

— RESPONSE STRUCTURE —
All your responses must be:
- Clear, readable, and well-organized.
- Use **headings**, **paragraphs**, and **bullet points** to explain complex topics.
- Provide **cause and effect**, **social context**, and **historical significance**, not just raw facts.

— LANGUAGE —
You must always reply in the **same language** the user used in their question (Arabic or English). For example:
- If the user asks in Arabic: "من هو محمد علي؟" → your reply must be in Arabic.
- If the user asks in English: "Who was Muhammad Ali?" → your reply must be in English.

— OUT-OF-SCOPE TOPICS —
You do NOT answer questions about:
- Current events after 2011
- Modern politics or current political figures
- Personal advice or opinions on contemporary matters
- Topics unrelated to Egyptian history
- Cooking, technology, or other non-historical subjects

For such queries, politely redirect: "I am The Chronicler of the Nile, focused exclusively on Egyptian history through 2011. Perhaps you'd like to explore a fascinating period from Egypt's rich past instead?"
"""

OFF_TOPIC_RESPONSE = "I am The Chronicler of the Nile, specializing exclusively in Egyptian history across all eras. I cannot assist with topics outside Egyptian history. Please ask me about pharaohs, dynasties, monuments, Islamic Egypt, modern Egyptian history, or any aspect of Egypt's rich historical heritage."

CURRENT_EVENTS_RESPONSE = "My historical records conclude in 2011. I cannot provide information about more recent events."

# Topic filtering functions
def is_egypt_history_related(message: str) -> bool:
    """Check if the message is related to Egyptian history"""
    egypt_keywords = [
        # Ancient Egypt
        'pharaoh', 'dynasty', 'pyramid', 'sphinx', 'mummy', 'hieroglyph', 'papyrus', 'nile',
        'egypt', 'egyptian', 'cairo', 'alexandria', 'luxor', 'aswan', 'giza', 'karnak',
        'temple', 'tomb', 'sarcophagus', 'ankh', 'osiris', 'isis', 'horus', 'ra', 'anubis',
        'tutankhamun', 'ramses', 'cleopatra', 'akhenaten', 'hatshepsut', 'thutmose',
        
        # Islamic/Medieval Egypt
        'mamluk', 'fatimid', 'ayyubid', 'saladin', 'mosque', 'minaret', 'islamic egypt',
        'arab conquest', 'amr ibn al-as', 'fustat', 'sultan', 'caliph',
        
        # Modern Egypt
        'muhammad ali', 'khedive', 'ottoman egypt', 'british occupation', 'suez canal',
        'nasser', 'sadat', 'mubarak', '1952 revolution', '1919 revolution', 'wafd',
        
        # Geographic
        'upper egypt', 'lower egypt', 'delta', 'sinai', 'red sea', 'mediterranean',
        
        # General historical terms in context
        'coptic', 'nubia', 'ptolemy', 'alexander', 'roman egypt'
    ]
    
    # Arabic keywords
    arabic_keywords = [
        'مصر', 'المصري', 'فرعون', 'هرم', 'نيل', 'قاهرة', 'اسكندرية', 'الأقصر',
        'أسوان', 'معبد', 'مقبرة', 'توت عنخ آمون', 'رمسيس', 'كليوباترا',
        'محمد علي', 'ناصر', 'السادات', 'مبارك', 'ثورة', 'مملوك', 'فاطمي',
        'أيوبي', 'صلاح الدين', 'مسجد', 'سلطان', 'خليفة', 'قبطي'
    ]
    
    message_lower = message.lower()
    
    # Check English keywords
    for keyword in egypt_keywords:
        if keyword in message_lower:
            return True
    
    # Check Arabic keywords
    for keyword in arabic_keywords:
        if keyword in message:
            return True
    
    # Check for historical question patterns about Egypt
    egypt_patterns = [
        r'egypt\w*\s+(history|historical|ancient|modern)',
        r'(history|historical)\s+.*egypt',
        r'(pharaoh|pyramid|nile).*egypt',
        r'egypt.*\b(dynasty|kingdom|period|era)\b'
    ]
    
    for pattern in egypt_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

def is_current_events_query(message: str) -> bool:
    """Check if message is asking about current events or politics after 2011"""
    current_events_keywords = [
        'current president', 'today', 'now', 'recent', 'latest', 'current government',
        'el sisi', 'sisi', '2012', '2013', '2014', '2015', '2016', '2017', '2018', 
        '2019', '2020', '2021', '2022', '2023', '2024', '2025',
        'current situation', 'nowadays', 'present', 'contemporary'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in current_events_keywords)

# Session state initialization
def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False

def configure_gemini():
    """Configure Google Gemini API"""
    try:
        # Try to get API key from Streamlit secrets first, then environment
        api_key = None
        
        # Check Streamlit secrets (for Hugging Face deployment)
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            # Fall back to environment variable
            api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            st.session_state.gemini_configured = True
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False

def format_conversation_for_gemini(history: List[Dict]) -> List[Dict]:
    """Convert conversation history to Gemini format"""
    formatted_history = []
    for msg in history:
        if msg.get('role') == 'system':
            continue  # Skip system messages for Gemini format
        role = "user" if msg.get('role') == "user" else "model"
        formatted_history.append({
            "role": role,
            "parts": [msg.get('content', '')]
        })
    return formatted_history

def manage_conversation_length(history: List[Dict], max_length: int = 20) -> List[Dict]:
    """Manage conversation history length"""
    if len(history) <= max_length:
        return history
    
    system_messages = [msg for msg in history[:2] if msg.get('role') == 'system']
    recent_messages = history[-(max_length - len(system_messages)):]
    return system_messages + recent_messages

def send_message_to_gemini(message: str, history: List[Dict]) -> str:
    """Send message to Gemini API and get response"""
    try:
        # Manage conversation history
        managed_history = manage_conversation_length(history)
        
        # Add system prompt if this is the start of conversation
        if not managed_history:
            system_message = {
                'role': 'system',
                'content': CHRONICLER_SYSTEM_PROMPT,
                'timestamp': datetime.now().isoformat()
            }
            managed_history.append(system_message)
        
        # Check for current events queries first
        if is_current_events_query(message):
            return CURRENT_EVENTS_RESPONSE
        
        # Check if the message is related to Egyptian history
        if not is_egypt_history_related(message):
            return OFF_TOPIC_RESPONSE
        
        # Format conversation for Gemini (exclude system messages and the current user message)
        gemini_history = format_conversation_for_gemini(managed_history)
        
        # Initialize Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
                candidate_count=1
            )
        )
        
        # Start chat with history
        chat = model.start_chat(history=gemini_history)
        
        # Send message and get response
        response = chat.send_message(message)
        
        return response.text
        
    except Exception as e:
        return f"I apologize, but I encountered an error while consulting my records: {str(e)}"

def reset_conversation():
    """Reset the conversation"""
    st.session_state.conversation_history = []
    st.success("Conversation reset successfully!")

def display_message(role: str, content: str, timestamp: str = None):
    """Display a chat message"""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
            if timestamp:
                st.caption(f"Sent: {datetime.fromisoformat(timestamp).strftime('%H:%M:%S')}")
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🏺"):
            st.write(content)
            if timestamp:
                st.caption(f"Replied: {datetime.fromisoformat(timestamp).strftime('%H:%M:%S')}")

def main():
    initialize_session_state()
    
    # Configure Gemini API
    if not st.session_state.gemini_configured:
        if not configure_gemini():
            st.error("⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to the secrets.")
            st.info("To deploy on Hugging Face Spaces, add your API key in the Space settings under 'Secrets'.")
            st.stop()
    
    # Header
    st.title("🏺 The Chronicler of the Nile")
    st.markdown("*Your Guide Through the Millennia of Egyptian History*")
    
    # Sidebar
    with st.sidebar:
        st.header("📜 About The Chronicler")
        st.markdown("""
        The Chronicler of the Nile is your sophisticated AI companion for exploring Egyptian history across all periods:
        
        **🏛️ Ancient Egypt**
        - Pharaohs and dynasties
        - Gods and mythology
        - Monuments and pyramids
        
        **🏺 Graeco-Roman Period**
        - Ptolemaic dynasty
        - Cleopatra and Roman rule
        - Early Christianity
        
        **🕌 Islamic Egypt**
        - Arab conquest
        - Fatimid, Ayyubid, Mamluk periods
        - Cultural developments
        
        **🏛️ Ottoman & Modern Egypt**
        - Turkish rule
        - Muhammad Ali dynasty
        - Revolution and modern history
        """)
        
        st.divider()
        
        # API Status
        if st.session_state.gemini_configured:
            st.success("✅ Gemini API Connected")
        else:
            st.error("❌ API Not Configured")
        
        st.divider()
        
        # Conversation controls
        if st.button("🔄 Reset Conversation", type="secondary"):
            reset_conversation()
            st.rerun()
        
        # Conversation stats
        if st.session_state.conversation_history:
            user_msgs = len([msg for msg in st.session_state.conversation_history if msg.get('role') == 'user'])
            st.metric("Messages Exchanged", user_msgs)
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        messages_to_display = [msg for msg in st.session_state.conversation_history if msg.get('role') != 'system']
        
        if not messages_to_display:
            st.info("👋 Greetings! I am the Chronicler of the Nile. Ask me anything about Egyptian history, from the age of the pharaohs to modern times.")
        
        for message in messages_to_display:
            display_message(
                message.get('role'),
                message.get('content'),
                message.get('timestamp')
            )
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_cols = st.columns(2)
    
    with example_cols[0]:
        if st.button("🏺 Tell me about Cleopatra VII"):
            st.session_state.example_question = "Tell me about Cleopatra VII and her role in Egyptian and Roman history."
        if st.button("🏛️ How were the pyramids built?"):
            st.session_state.example_question = "How were the pyramids of Giza constructed and what was their purpose?"
    
    with example_cols[1]:
        if st.button("🕌 What was the Islamic conquest of Egypt?"):
            st.session_state.example_question = "Describe the Islamic conquest of Egypt and its impact on the region."
        if st.button("🏛️ Tell me about the 1952 Revolution"):
            st.session_state.example_question = "What led to the 1952 Egyptian Revolution and what were its consequences?"
    
    # Handle example questions
    if 'example_question' in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
        
        # Process the example question
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.conversation_history.append(user_message)
        
        with st.chat_message("assistant", avatar="🏺"):
            with st.spinner("The Chronicler is consulting the ancient records..."):
                response = send_message_to_gemini(user_input, st.session_state.conversation_history)
                st.write(response)
                
                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(assistant_message)
        
        st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask the Chronicler about Egyptian history...")
    
    if user_input:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.conversation_history.append(user_message)
        
        # Get and display assistant response
        with st.chat_message("assistant", avatar="🏺"):
            with st.spinner("The Chronicler is consulting the ancient records..."):
                response = send_message_to_gemini(user_input, st.session_state.conversation_history)
                st.write(response)
                
                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(assistant_message)
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main()

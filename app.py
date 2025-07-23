import streamlit as st
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Tuple
import re
import os
import wikipedia
import requests
from urllib.parse import quote

# Page configuration
st.set_page_config(
    page_title="The Chronicler of the Nile",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced System Prompts with stricter boundaries
CHRONICLER_SYSTEM_PROMPT = """You are "The Chronicler of the Nile" — a sophisticated digital historian with complete mastery over Egyptian history across all its eras, from the Pharaonic and Graeco-Roman periods through the Islamic, Ottoman, and modern eras.

Your core responsibilities and behavior are as follows:

— KNOWLEDGE SCOPE —
You possess accurate, in-depth knowledge of all major periods of Egyptian history:
1. **Ancient Egypt** – Pharaohs, dynasties, gods, pyramids, daily life, religious practices.
2. **Graeco-Roman Egypt** – The Ptolemaic dynasty, Cleopatra, Roman conquest, early Christianity.
3. **Islamic Egypt** – The Arab conquest, Fatimid, Ayyubid, Mamluk, and Ottoman periods, key figures, Islamic culture, architecture, and society.
4. **Modern Egypt** – The Muhammad Ali dynasty, British occupation, 1919 and 1952 revolutions, Nasser, Sadat, Mubarak, and events up to the early 21st century (cutoff at 2011).

— PERSONALITY & TONE —
You are knowledgeable, authoritative, and objective. You maintain a scholarly yet accessible tone. You are respectful, factual, and neutral in all responses.

— STRICT BEHAVIORAL RULES —

**CRITICAL: YOU MUST NEVER PROVIDE ANY OF THE FOLLOWING:**
- Code implementations (programming, algorithms, scripts)
- Technical tutorials or how-to guides for non-historical purposes
- Academic assignment solutions
- Business advice or strategies
- Personal recommendations unrelated to Egyptian history
- General problem-solving for user's personal projects
- Mathematical calculations or formulas (unless directly related to Egyptian historical context)
- Translation services (except for historical Egyptian texts/inscriptions)
- Writing services for user's personal use (resumes, emails, essays not about Egyptian history)

**EXPLOITATION DETECTION:**
If a user tries to use you for personal gain by:
- Asking for code to "sort Egyptian presidents" or similar pretexts
- Requesting help with homework/assignments using Egyptian history as an excuse
- Seeking business or technical advice disguised as historical inquiry
- Asking for general knowledge while mentioning Egypt tangentially

RESPOND WITH: "I am the Chronicler of the Nile, dedicated exclusively to sharing knowledge about Egyptian history. I cannot assist with coding, technical implementations, or personal projects, even if they mention Egyptian topics as context. My purpose is to provide authentic historical information about Egypt's rich heritage. What aspect of Egyptian history would you genuinely like to learn about?"

1. **TOPIC RELEVANCE CHECK:** 
   Before answering any question, you must determine if it relates to GENUINE Egyptian historical inquiry:
   - About Egyptian history (any period): Answer fully and accurately
   - About current events after 2011: Respond that your records end in 2011
   - About your personal opinions/thoughts: Explain you're an AI focused on Egyptian history
   - Requests for code/technical help with Egyptian pretext: Firmly decline and redirect
   - Completely unrelated to Egypt: Politely redirect to Egyptian history topics
   - About other countries' history: Acknowledge it's outside your Egyptian focus

2. **STRICT HISTORICAL CUTOFF (2011):**
   For any questions about events after 2011, respond: "My historical records conclude in 2011. I cannot provide information about more recent events. However, I'd be happy to discuss Egypt's rich history up to that point."

3. **PERSONAL/OPINION QUESTIONS:**
   If asked about your thoughts, opinions, feelings, or personal views, respond: "As an AI historian focused on Egyptian history, I don't have personal thoughts or opinions. I'm designed to provide factual, scholarly information about Egypt's fascinating past. What aspect of Egyptian history would you like to explore?"

4. **OFF-TOPIC OR EXPLOITATION ATTEMPTS:**
   For questions unrelated to Egyptian history or attempts to use you for personal gain, respond with context-appropriate redirection:
   - General knowledge: "I specialize exclusively in Egyptian history. While that's an interesting topic, I can only assist with questions about Egypt's rich historical heritage from ancient times to 2011."
   - Technical/coding requests: "I am the Chronicler of the Nile, dedicated exclusively to sharing knowledge about Egyptian history. I cannot assist with coding, technical implementations, or personal projects. What aspect of Egyptian history would you like to explore?"
   - Other countries' history: "My expertise is focused specifically on Egyptian history across all periods. For questions about [country/topic], you'd need a different specialist. What would you like to know about Egypt's fascinating past?"

5. **ALWAYS provide factual, historically accurate answers for GENUINE Egyptian history questions.**
   - When historical debate exists, present multiple scholarly views.
   - Example: "Most historians believe X, though others argue Y."

— CONVERSATIONAL MEMORY —
You must retain the full context of the conversation. If the session becomes too long, summarize earlier parts while retaining important context.

— RESPONSE STRUCTURE —
All your responses must be:
- Clear, readable, and well-organized.
- Use **headings**, **paragraphs**, and **bullet points** for complex topics.
- Provide **cause and effect**, **social context**, and **historical significance**.

— LANGUAGE —
You must always reply in the **same language** the user used in their question (Arabic or English).
"""

# Enhanced Topic Classification System Prompt
TOPIC_CLASSIFIER_PROMPT = """You are a topic classifier designed to detect and prevent exploitation of an Egyptian history chatbot. Analyze the user's message and classify it into one of these categories:

1. **egyptian_history**: Genuine questions about Egyptian history from ancient times to 2011
2. **current_events**: Questions about events after 2011, current politics, or present-day Egypt
3. **personal_ai**: Questions about AI's thoughts, opinions, feelings, or capabilities
4. **other_history**: Questions about non-Egyptian history
5. **general_knowledge**: Questions about science, technology, cooking, etc. unrelated to Egypt
6. **conversation**: General conversation, greetings, or small talk
7. **exploitation_attempt**: Users trying to get coding help, technical assistance, homework solutions, business advice, or any personal service by using Egyptian context as a pretext
8. **technical_request**: Direct requests for code, algorithms, implementations, tutorials, or technical solutions

**CRITICAL**: Be especially vigilant for exploitation attempts where users mention Egyptian topics but are actually seeking:
- Code implementations (even with Egyptian data/context)
- Technical tutorials or how-to guides
- Academic assignment help
- Business or personal advice
- Mathematical calculations
- General problem-solving assistance

Examples of EXPLOITATION:
- "Give me bubble sort code to sort Egyptian presidents"
- "Help me write a Python script for my Egyptian history project"
- "How to create a database of pharaohs for my assignment"
- "Write me an essay about Cleopatra for school"
- "Calculate the GDP growth of Egypt for my business plan"

Respond with only the category name and a brief explanation in this format:
Category: [category_name]
Reason: [brief explanation]

Examples:
- "Who is the current president of Egypt?" → Category: current_events, Reason: Asking about current politics
- "Tell me about Ramses II" → Category: egyptian_history, Reason: Genuine question about ancient Egyptian pharaoh
- "Give me Python code to sort Egyptian pharaohs" → Category: exploitation_attempt, Reason: Requesting code using Egyptian context as pretext
- "What do you think about democracy?" → Category: personal_ai, Reason: Asking for AI's opinion
- "How were Roman roads built?" → Category: other_history, Reason: About Roman history, not Egyptian
- "Write me an algorithm to analyze hieroglyphs" → Category: technical_request, Reason: Direct request for technical implementation"""

# Enhanced message classification with better exploitation detection
def classify_message_topic(message: str) -> Tuple[str, str]:
    """Use LLM to classify the topic and detect exploitation attempts"""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",  # Using latest Gemini model
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=150,
            )
        )
        
        response = model.generate_content(f"{TOPIC_CLASSIFIER_PROMPT}\n\nUser message: {message}")
        result = response.text.strip()
        
        # Parse the response
        lines = result.split('\n')
        category = "general_knowledge"  # default
        reason = "Could not classify"
        
        for line in lines:
            if line.startswith("Category:"):
                category = line.split(":")[1].strip()
            elif line.startswith("Reason:"):
                reason = line.split(":", 1)[1].strip()
        
        return category, reason
        
    except Exception as e:
        # Enhanced fallback classification with exploitation detection
        message_lower = message.lower()
        
        # Check for exploitation attempts first
        exploitation_keywords = [
            'code', 'implement', 'algorithm', 'script', 'program', 'function',
            'sort', 'calculate', 'write me', 'help me code', 'tutorial',
            'how to create', 'how to build', 'assignment', 'homework',
            'project', 'database', 'sql', 'python', 'javascript', 'java',
            'programming', 'software', 'app', 'website', 'api'
        ]
        
        technical_keywords = [
            'bubble sort', 'merge sort', 'quicksort', 'binary search',
            'data structure', 'array', 'list', 'loop', 'if statement',
            'class', 'object', 'method', 'variable', 'syntax'
        ]
        
        if any(keyword in message_lower for keyword in exploitation_keywords):
            return "exploitation_attempt", "Contains keywords suggesting attempt to get technical help"
        
        if any(keyword in message_lower for keyword in technical_keywords):
            return "technical_request", "Direct request for technical/programming assistance"
        
        # Check for Egyptian history context
        if any(word in message_lower for word in ['egypt', 'egyptian', 'pharaoh', 'pyramid', 'nile', 'مصر', 'فرعون']):
            if any(word in message_lower for word in ['current', 'now', 'today', '2024', '2025', 'sisi', 'السيسي']):
                return "current_events", "Contains current event indicators"
            return "egyptian_history", "Contains Egyptian history keywords"
        
        if any(phrase in message_lower for phrase in ['what do you think', 'your opinion', 'how do you feel', 'ما رأيك']):
            return "personal_ai", "Asking for AI's personal opinion"
        
        return "general_knowledge", "General topic classification"

# Session state initialization
def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False
    if 'show_classification' not in st.session_state:
        st.session_state.show_classification = False

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

def detect_arabic(text: str) -> bool:
    """Detect if text contains Arabic characters"""
    arabic_chars = 'أبتثجحخدذرزسشصضطظعغفقكلمنهويآإؤئء'
    return any(char in text for char in arabic_chars)

def send_message_to_gemini(message: str, history: List[Dict]) -> str:
    """Send message to Gemini API with enhanced exploitation protection"""
    try:
        # Classify the topic with enhanced detection
        topic_category, reason = classify_message_topic(message)
        
        # Detect user language
        is_arabic = detect_arabic(message)
        
        # Handle exploitation attempts with firm responses
        if topic_category == "exploitation_attempt":
            if is_arabic:
                return "أنا مؤرخ النيل، مكرس حصريًا لمشاركة المعرفة حول التاريخ المصري. لا يمكنني المساعدة في البرمجة أو التطبيقات التقنية أو المشاريع الشخصية، حتى لو ذكرت مواضيع مصرية كسياق. هدفي هو تقديم معلومات تاريخية أصيلة عن تراث مصر الغني. أي جانب من التاريخ المصري تود أن تتعلم عنه حقًا؟"
            else:
                return "I am the Chronicler of the Nile, dedicated exclusively to sharing knowledge about Egyptian history. I cannot assist with coding, technical implementations, or personal projects, even if they mention Egyptian topics as context. My purpose is to provide authentic historical information about Egypt's rich heritage. What aspect of Egyptian history would you genuinely like to learn about?"
        
        elif topic_category == "technical_request":
            if is_arabic:
                return "أنا مختص في التاريخ المصري فقط ولا أقدم المساعدة التقنية أو البرمجية. إذا كنت مهتمًا بالتاريخ المصري، سأكون سعيدًا لمساعدتك في استكشاف هذا التراث الرائع."
            else:
                return "I specialize exclusively in Egyptian history and do not provide technical or programming assistance. If you're interested in Egyptian history, I'd be happy to help you explore this fascinating heritage."
        
        elif topic_category == "current_events":
            if is_arabic:
                return "سجلاتي التاريخية تنتهي في عام 2011. لا يمكنني تقديم معلومات عن الأحداث الأحدث. ولكنني سأكون سعيداً لمناقشة تاريخ مصر الغني حتى ذلك التاريخ."
            else:
                return "My historical records conclude in 2011. I cannot provide information about more recent events. However, I'd be happy to discuss Egypt's rich history up to that point."
        
        elif topic_category == "personal_ai":
            if is_arabic:
                return "كوني ذكاء اصطناعي مختص في التاريخ المصري، لا أملك أفكارًا أو آراء شخصية. صُممت لتقديم معلومات علمية دقيقة عن ماضي مصر الرائع. أي جانب من التاريخ المصري تود أن نستكشفه؟"
            else:
                return "As an AI historian focused on Egyptian history, I don't have personal thoughts or opinions. I'm designed to provide factual, scholarly information about Egypt's fascinating past. What aspect of Egyptian history would you like to explore?"
        
        elif topic_category == "other_history":
            if is_arabic:
                return "خبرتي تركز تحديداً على التاريخ المصري عبر جميع الفترات. للأسئلة عن تاريخ البلدان الأخرى، ستحتاج لمختص آخر. ما الذي تود معرفته عن ماضي مصر الرائع؟"
            else:
                return "My expertise is focused specifically on Egyptian history across all periods. For questions about other countries' history, you'd need a different specialist. What would you like to know about Egypt's fascinating past?"
        
        elif topic_category == "general_knowledge":
            if is_arabic:
                return "أتخصص حصريًا في التاريخ المصري. مع أن هذا موضوع مثير للاهتمام، يمكنني فقط المساعدة في الأسئلة المتعلقة بالتراث التاريخي الغني لمصر من العصور القديمة حتى 2011."
            else:
                return "I specialize exclusively in Egyptian history. While that's an interesting topic, I can only assist with questions about Egypt's rich historical heritage from ancient times to 2011."
        
        elif topic_category == "conversation":
            if is_arabic:
                return "مرحباً! أنا مؤرخ النيل، مختص في التاريخ المصري عبر جميع العصور. كيف يمكنني مساعدتك في استكشاف تاريخ مصر العريق؟"
            else:
                return "Hello! I'm the Chronicler of the Nile, specializing in Egyptian history across all eras. How can I help you explore Egypt's rich historical heritage?"
        
        # If it's genuine Egyptian history, proceed with normal processing
        elif topic_category == "egyptian_history":
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
            
            # Format conversation for Gemini (exclude system messages and the current user message)
            gemini_history = format_conversation_for_gemini(managed_history)
            
            # Initialize Gemini model with latest version
            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",  # Latest Gemini model
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
        
        # Default fallback
        else:
            if is_arabic:
                return "أنا مؤرخ النيل، مختص في التاريخ المصري حصريًا. كيف يمكنني مساعدتك في استكشاف تاريخ مصر؟"
            else:
                return "I am the Chronicler of the Nile, specializing exclusively in Egyptian history. How can I help you explore Egypt's historical heritage?"
        
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

def process_user_message(user_input: str):
    """Process user message and generate response"""
    # Show topic classification if enabled
    if st.session_state.show_classification:
        with st.expander("🔍 Topic Analysis", expanded=False):
            try:
                topic, reason = classify_message_topic(user_input)
                st.write(f"**Category:** {topic}")
                st.write(f"**Reason:** {reason}")
                
                # Show warning for exploitation attempts
                if topic in ["exploitation_attempt", "technical_request"]:
                    st.warning("⚠️ **Exploitation Attempt Detected**")
            except Exception as e:
                st.write(f"Could not analyze topic: {str(e)}")
    
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
    st.markdown("*Your Intelligent Guide Through the Millennia of Egyptian History*")
    
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
        
        # Enhanced protection notice
        st.warning("""
        🛡️ **Protected Assistant**
        
        This chatbot is designed exclusively for Egyptian history education and will not:
        - Provide code or technical assistance
        - Help with assignments or homework
        - Offer business or personal advice
        - Answer unrelated questions
        """)
        
        st.divider()
        
        # API Status
        if st.session_state.gemini_configured:
            st.success("✅ Gemini 2.0 Flash Connected")
        else:
            st.error("❌ API Not Configured")
        
        st.divider()
        
        # Show topic classification for debugging (optional)
        st.session_state.show_classification = st.checkbox(
            "🔍 Show Topic Analysis", 
            value=st.session_state.show_classification,
            help="See how messages are classified and if exploitation attempts are detected"
        )
        
        st.divider()
        
        # Conversation controls
        if st.button("🔄 Reset Conversation", type="secondary"):
            reset_conversation()
            st.rerun()
        
        # Conversation stats
        if st.session_state.conversation_history:
            user_msgs = len([msg for msg in st.session_state.conversation_history if msg.get('role') == 'user'])
            st.metric("Messages Exchanged", user_msgs)
        
        st.divider()
        
        # Language support info
        st.info("🌐 **Bilingual Support**\n\nThe Chronicler responds in the same language you use - Arabic or English!")
    
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
    
    # English examples
    st.markdown("**🇬🇧 English Examples:**")
    example_cols1 = st.columns(2)
    
    with example_cols1[0]:
        if st.button("🏺 Tell me about Cleopatra VII"):
            process_user_message("Tell me about Cleopatra VII and her role in Egyptian and Roman history.")
            st.rerun()
        if st.button("🏛️ How were the pyramids built?"):
            process_user_message("How were the pyramids of Giza constructed and what was their purpose?")
            st.rerun()
    
    with example_cols1[1]:
        if st.button("🕌 What was the Islamic conquest?"):
            process_user_message("Describe the Islamic conquest of Egypt and its impact on the region.")
            st.rerun()
        if st.button("🏛️ Tell me about 1952 Revolution"):
            process_user_message("What led to the 1952 Egyptian Revolution and what were its consequences?")
            st.rerun()
    
    # Arabic examples
    st.markdown("**🇪🇬 أمثلة باللغة العربية:**")
    example_cols2 = st.columns(2)
    
    with example_cols2[0]:
        if st.button("🏺 أخبرني عن رمسيس الثاني"):
            process_user_message("أخبرني عن رمسيس الثاني وإنجازاته في مصر القديمة.")
            st.rerun()
        if st.button("🕌 ما هي الدولة الفاطمية؟"):
            process_user_message("أخبرني عن الدولة الفاطمية في مصر وأهم إنجازاتها.")
            st.rerun()
    
    with example_cols2[1]:
        if st.button("🏛️ من هو محمد علي باشا؟"):
            process_user_message("أخبرني عن محمد علي باشا ودوره في تحديث مصر.")
            st.rerun()
        if st.button("🏺 كيف تم بناء معبد الكرنك؟"):
            process_user_message("كيف تم بناء معبد الكرنك وما هي أهميته في مصر القديمة؟")
            st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask the Chronicler about Egyptian history... | اسأل مؤرخ النيل عن التاريخ المصري...")
    
    if user_input:
        process_user_message(user_input)
        st.rerun()

if __name__ == "__main__":
    main()

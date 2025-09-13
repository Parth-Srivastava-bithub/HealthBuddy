# AI-Guided First-Aid Support Chatbot (Streamlit App)
# This script creates an advanced web interface for a first-aid chatbot.
# It incorporates a dual-LLM approach: Gemini for intent summarization and Groq for RAG-based answers.
# Features include local embeddings, a persistent FAISS vector store, an enhanced dark theme,
# improved conversational memory, and a user-controlled "Deep Think" feature.

import os
import time
from datetime import date
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
FAISS_INDEX_PATH = "faiss_first_aid_index"
MEMORY_WINDOW = 3 # Number of past messages to consider as context

# --- GAMIFICATION CONFIG ---
LEVELS = {
    1: {"name": "Beginner", "threshold": 0},
    2: {"name": "Learner", "threshold": 100},
    3: {"name": "Practitioner", "threshold": 250},
    4: {"name": "Guardian", "threshold": 500},
    5: {"name": "Expert", "threshold": 1000}
}
POINTS_CONFIG = {
    'on_topic_query': 15,
    'bleeding_button': 10,
    'burn_button': 10,
    'anxiety_button': 20
}
BUTTON_COOLDOWN_SECONDS = 300  # 5 minutes
DAILY_POINT_CAP = 100
ON_TOPIC_KEYWORDS = [
    'cut', 'scrape', 'bleed', 'burn', 'scald', 'anxiety', 'panic',
    'stress', 'help', 'what', 'how', 'treat', 'should i', 'steps',
    'wound', 'injury', 'pain', 'breathe', 'dizzy'
]

# --- FUNCTIONS ---

@st.cache_resource
def load_embeddings_model():
    """Load the embeddings model, cached for performance."""
    print("Initializing local embeddings model (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store(_embeddings):
    """Load FAISS index from disk if it exists, otherwise create and save it."""
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        return FAISS.load_local(FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating a new one...")
        docs = [
            Document(
                page_content="Minor Cuts and Scrapes: First, wash your hands to avoid infection. Gently clean the wound with cool water and mild soap. Apply gentle pressure with a clean cloth to stop the bleeding. Once bleeding stops, apply a thin layer of antibiotic ointment and cover it with a sterile bandage. Change the bandage daily.",
                metadata={"source": "basic_first_aid.txt"}
            ),
            Document(
                page_content="Minor Burn: Cool the burn immediately with cool (not cold) running water for about 10 minutes or until the pain lessens. Cover the burn with a sterile gauze bandage (not fluffy cotton). Do not use ice, ointments, or butter, which can cause more damage.",
                metadata={"source": "basic_burn_care.txt"}
            ),
            Document(
                page_content="Emotional Stress and Anxiety Attack: If you're feeling overwhelmed, try deep breathing exercises. Inhale slowly for 4 seconds, hold for 4 seconds, and exhale slowly for 6 seconds. Repeat this several times. Grounding techniques can also help: name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. Reassure yourself that this feeling will pass.",
                metadata={"source": "mental_health_support.txt"}
            ),
            Document(
                page_content="Recognizing Serious Situations: If a cut is deep, spurting blood, or doesn't stop bleeding after 10 minutes of pressure, seek immediate medical attention. For emotional distress, if you have thoughts of harming yourself or others, it is crucial to speak with a professional immediately.",
                metadata={"source": "emergency_referrals.txt"}
            ),
            Document(
                page_content="Disclaimer: This is an AI-powered first-aid assistant. The guidance provided is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.",
                metadata={"source": "disclaimer.txt"}
            )
        ]
        vector_store = FAISS.from_documents(docs, embedding=_embeddings)
        print(f"Saving new FAISS index to {FAISS_INDEX_PATH}...")
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store

def get_conversational_rag_chain(_groq_llm, _vector_store):
    """Create and return the full RAG chain with the First-Aid Buddy persona."""
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_system_prompt = (
        "Given a chat history and the latest user question "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(_groq_llm, retriever, retriever_prompt)

    qa_system_prompt = (
        "You are a chill AF buddy who’s here for first-aid and stress stuff. "
        "Talk like a real bro texting another bro—super casual, short, and to the point. "
        "Use emojis like ❤️‍🩹, 🩹, 🚑, 🤯, but don't overdo it. Keep it real and friendly."
        "Give safe, clear advice but don’t sound like a textbook. Make instructions simple and step-by-step."
        "\n\n"
        "--- IMPORTANT RULES ---\n"
        "1. **MANDATORY RESPONSE STRUCTURE:** You MUST follow this structure for every answer:\n"
        "   - **One-Liner Action:** Start with a short, direct instruction. What's the first thing to do?\n"
        "   - **Simple Steps:** Provide a clear, step-by-step guide based on the context.\n"
        "   - **When to get help:** If it sounds serious (like heavy bleeding, severe burns, or suicidal thoughts), clearly tell them to call a professional or emergency services ASAP. Still say it like a concerned friend.\n"
        "2. **Stay on Topic:** If the user asks something unrelated to first-aid or mental well-being, gently guide them back. E.g., 'Yo, interesting question, but let's stay focused on the first-aid stuff for now, yeah?'\n"
        "3. **No Hallucination:** ONLY use the information provided in the 'Context' below. Do not invent medical advice.\n"
        "4. **Be a Buddy:** Use words like 'bro', 'man', 'okay so...', 'make sure you...'. Avoid being formal."
        "\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(_groq_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_intent_summary_from_gemini(user_query, gemini_client, retries=2, delay=1.5):
    """
    Use Gemini to generate a short 'about text' from the user's query.
    Enhanced with retry logic, clean formatting, and fallbacks.
    """
    system_prompt = (
        "You are an intent summarizer for a first-aid chatbot. "
        "Take the user's raw input and generate a short 'about text' (1–2 sentences) "
        "that describes:\n"
        "- What physical or emotional state the user is in.\n"
        "- What kind of first-aid information would help them.\n"
        "Respond with plain text only. No JSON, no formatting, no bullet points."
    )
    
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    for attempt in range(1, retries + 1):
        try:
            resp = gemini_client.invoke(msgs)
            summary = resp.content.strip()

            summary = " ".join(summary.split())
            if not summary:
                raise ValueError("Empty response from Gemini.")
            return summary
        except Exception as e:
            print(f"[Gemini Attempt {attempt}] Error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                return "Could not clearly analyze the user's situation."

def update_long_term_summary(groq_llm, previous_summary, recent_history):
    """Updates the long-term summary of the conversation using Groq."""
    print("Updating long-term memory...")

    # Format the recent history for the prompt
    formatted_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Buddy'}: {m.content}" for m in recent_history])

    summarizer_prompt = (
        "You are a conversation memory updater. Your task is to maintain a short episodic memory of the chat. "
        "Combine the previous summary with the recent messages and return an updated summary. "
        "The updated summary must be only 3–4 lines, concise, and focused on key facts, actions, or emotions. "
        "Do not add extra advice, only preserve context."
    )
    
    summarizer_msgs = [
        SystemMessage(content=summarizer_prompt),
        HumanMessage(content=f"Previous Summary:\n{previous_summary}\n\nRecent Messages:\n{formatted_history}\n\nReturn the new summary:")
    ]

    try:
        response = groq_llm.invoke(summarizer_msgs)
        new_summary = response.content.strip()
        if new_summary:
            st.session_state.long_term_summary = new_summary
            st.toast("Memory updated successfully! 🧠", icon="✅")
            print(f"New Summary: {new_summary}")
    except Exception as e:
        print(f"Error updating long-term memory: {e}")
        st.toast("Could not update memory.", icon="❌")


def get_current_level(score):
    """Determines the user's current level based on their score."""
    current_level = 1
    for level, data in LEVELS.items():
        if score >= data["threshold"]:
            current_level = level
        else:
            break
    return current_level

def update_score(action_type, query=""):
    """Handles all gamification logic for the preparedness level."""
    if st.session_state.last_active_date != date.today():
        st.session_state.daily_points_earned = 0
        st.session_state.last_active_date = date.today()

    if st.session_state.daily_points_earned >= DAILY_POINT_CAP:
        st.toast(f"You've hit your daily cap of {DAILY_POINT_CAP} points. Come back tomorrow for more learning! 🗓️", icon="✅")
        return

    points_to_add = 0
    is_on_cooldown = False

    if "button" in action_type:
        now = time.time()
        last_press = st.session_state.last_button_press_time.get(action_type, 0)
        if now - last_press < BUTTON_COOLDOWN_SECONDS:
            is_on_cooldown = True
        else:
            st.session_state.last_button_press_time[action_type] = now
            points_to_add = POINTS_CONFIG.get(action_type, 0)
    
    elif action_type == 'on_topic_query':
        if any(keyword in query.lower() for keyword in ON_TOPIC_KEYWORDS):
            points_to_add = POINTS_CONFIG.get(action_type, 0)
        else:
            st.toast("Ask a health-related question to earn points! ✨", icon="💡")
            return

    if is_on_cooldown:
        st.toast("Patience is a virtue! The button is on cooldown.", icon="⏳")
        return

    if points_to_add > 0:
        points_to_add = min(points_to_add, DAILY_POINT_CAP - st.session_state.daily_points_earned)
        
        previous_level = get_current_level(st.session_state.score)
        st.session_state.score += points_to_add
        st.session_state.daily_points_earned += points_to_add
        new_level = get_current_level(st.session_state.score)

        st.toast(f"+{points_to_add} points for being prepared! ✨", icon="❤️‍🩹")
        
        if new_level > previous_level:
            st.toast(f"Level Up! You've reached: {LEVELS[new_level]['name']}!", icon="🎉")
            st.balloons()

def process_user_query(user_query, rag_chain, gemini_client, groq_llm):
    """
    Handles the chatbot response logic using the dual-LLM pipeline.
    Conditionally uses Gemini based on the 'Deep Think' toggle.
    """
    if not user_query: return

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.chat_message("user", avatar="👤").markdown(user_query)
    
    # --- CONTEXT BUILDING ---
    # Combine long-term summary with recent history for a richer context
    context_messages = []
    if st.session_state.long_term_summary:
        summary_message = (
            "Key points from earlier in our chat:\n"
            f"{st.session_state.long_term_summary}"
        )
        context_messages.append(SystemMessage(content=summary_message))
    
    context_messages.extend(st.session_state.chat_history[-MEMORY_WINDOW:])

    with st.chat_message("assistant", avatar="❤️‍🩹"):
        with st.spinner("Getting the right info..."):
            
            # --- MODIFIED LOGIC ---
            # Check the "Deep Think" toggle from session state
            if st.session_state.get('deep_think_enabled', True):
                # "YES" PATH: Use Gemini to summarize
                # Step 1: Get intent summary from Gemini
                about_text = get_intent_summary_from_gemini(user_query, gemini_client)
                st.caption(f"*Assessing the situation: {about_text}*")

                # Step 2: Combine summary and query for Groq
                final_input = (
                    f"The following 'about text' describes the user’s situation.\n\n"
                    f"Use it to shape your reply while staying calm and direct.\n"
                    f"About text: {about_text}\n\n"
                    f"User’s actual query: {user_query}"
                )
            else:
                # "NO" PATH: Skip Gemini and use the raw query
                st.caption(f"*'Deep Think' is off. Going straight to the point...*")
                final_input = user_query
            
            # Step 3: Get final response from Groq RAG chain
            response = rag_chain.invoke({
                "input": final_input, 
                "chat_history": context_messages
            })
            ai_response = response.get('answer', "Man, my wires are crossed. Can you ask that a different way?")
            st.markdown(ai_response)
    
    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.session_state.message_counter += 2 # User + AI message = 2

    # --- MEMORY SUMMARIZATION LOGIC ---
    # Check if it's time to update the long-term summary
    if st.session_state.message_counter >= MEMORY_WINDOW and st.session_state.message_counter % MEMORY_WINDOW == 0:
        update_long_term_summary(
            groq_llm,
            st.session_state.long_term_summary,
            st.session_state.chat_history[-MEMORY_WINDOW:]
        )

# --- STREAMLIT APP ---
st.set_page_config(page_title="AI First-Aid Assistant", page_icon="❤️‍🩹", layout="centered")

# Custom CSS for a darker, modern theme
st.markdown("""
<style>
    /* Main App Body */
    .stApp {
        background-color: #0d1117; /* GitHub Dark-like background */
        background-image: linear-gradient(180deg, #0d1117 0%, #171426 100%);
        color: #c9d1d9; /* Light text for readability */
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 10rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }
    /* Main Title */
    h1 {
        color: #2ea043; /* Calm Green */
        text-align: center;
    }
    /* Sub-headers and text */
    h2, h3, .stMarkdown, .stButton > button p, .stCaption {
        color: #c9d1d9;
    }
    /* Buttons */
    [data-testid="stButton"] > button {
        background-color: #238636; /* Green accent */
        color: #ffffff;
        border: 1px solid #30363d;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Text Input */
    [data-testid="stChatInput"] {
        background-color: #010409;
    }
</style>
""", unsafe_allow_html=True)

st.title("❤️‍🩹 AI-Guided First-Aid Support")
st.markdown("<p style='text-align: center;'>I'm here for basic first-aid and emotional stress support. <br><b>This is not a substitute for professional medical advice. For emergencies, call for help.</b></p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    
    # --- NEW FEATURE: DEEP THINK TOGGLE ---
    st.toggle(
        "Enable 'Deep Think'", 
        value=st.session_state.get('deep_think_enabled', True), 
        key='deep_think_enabled',
        help="When ON, a second AI (Gemini) first summarizes your situation for a more detailed, contextual answer. When OFF, you get a faster, more direct response."
    )
    st.divider()

    if not groq_api_key or not gemini_api_key:
        st.info("Please enter both your Groq and Gemini API keys to begin.")
        st.stop()
    
    try:
        if "llms_initialized" not in st.session_state:
            st.session_state.groq_llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="gemma2-9b-it")
            st.session_state.gemini_client = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, max_output_tokens=256, google_api_key=gemini_api_key)
            st.session_state.llms_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize models. Check your API keys. Error: {e}")
        st.stop()

    st.header("🚨 Quick Access")
    if st.button("Minor Bleeding 🩸", use_container_width=True):
        update_score('bleeding_button')
        process_user_query("What are the first aid steps for a minor cut that's bleeding?", st.session_state.rag_chain, st.session_state.gemini_client, st.session_state.groq_llm)
    if st.button("Minor Burn 🔥", use_container_width=True):
        update_score('burn_button')
        process_user_query("How do I treat a minor burn?", st.session_state.rag_chain, st.session_state.gemini_client, st.session_state.groq_llm)
    if st.button("Anxiety Attack 😟", use_container_width=True):
        update_score('anxiety_button')
        process_user_query("I think I'm having an anxiety attack, what should I do?", st.session_state.rag_chain, st.session_state.gemini_client, st.session_state.groq_llm)
    st.divider()

    st.header("Your Preparedness Level")
    if "score" in st.session_state:
        score = st.session_state.score
        level = get_current_level(score)
        st.markdown(f"**Level {level}: {LEVELS[level]['name']}**")
        
        next_level = level + 1
        if next_level in LEVELS:
            threshold_current = LEVELS[level]['threshold']
            threshold_next = LEVELS[next_level]['threshold']
            progress = (score - threshold_current) / (threshold_next - threshold_current) if (threshold_next - threshold_current) > 0 else 1
            st.progress(progress, text=f"{score} / {threshold_next} points")
        else:
            st.progress(1.0, text="Max Level Reached! You're an Expert! 🎉")
        
        st.caption(f"Daily points: {st.session_state.get('daily_points_earned', 0)} / {DAILY_POINT_CAP}")

# Initialize session state for the first run
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "long_term_summary" not in st.session_state:
    st.session_state.long_term_summary = ""
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0
if "vector_store" not in st.session_state:
    embeddings = load_embeddings_model()
    st.session_state.vector_store = get_vector_store(embeddings)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_conversational_rag_chain(st.session_state.groq_llm, st.session_state.vector_store)

# Initialize Gamification State
if "score" not in st.session_state:
    st.session_state.score = 0
if "last_button_press_time" not in st.session_state:
    st.session_state.last_button_press_time = {}
if "daily_points_earned" not in st.session_state:
    st.session_state.daily_points_earned = 0
if "last_active_date" not in st.session_state:
    st.session_state.last_active_date = date.today()

# Display chat history from session state
for message in st.session_state.chat_history:
    avatar = "👤" if isinstance(message, HumanMessage) else "❤️‍🩹"
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant", avatar=avatar):
        st.markdown(message.content)

# Handle user input from chat box
text_input_query = st.chat_input("What's the situation?")
if text_input_query:
    update_score('on_topic_query', query=text_input_query)
    process_user_query(text_input_query, st.session_state.rag_chain, st.session_state.gemini_client, st.session_state.groq_llm)



# 🧠 AI First-Aid Chatbot Memory System

This document explains how the **chatbot memory** works for the AI-Guided First-Aid Support system.  
It handles both **short-term context** and **long-term episodic memory** to give smart, context-aware responses.

---

## 1️⃣ Short-Term Memory (Chat History Window)

- Maintains the **last few messages** (controlled by `MEMORY_WINDOW`).
- Default window: **3 messages**.
- Purpose: Provide immediate context for the RAG chain.
- Stored in `st.session_state.chat_history`.
- Example:
```python
st.session_state.chat_history = [
    HumanMessage(content="I have a minor cut."),
    AIMessage(content="Clean it with water first."),
    HumanMessage(content="Bleeding has stopped now.")
]
````

* Only **recent messages** are fed to the Groq chain during RAG.

---

## 2️⃣ Long-Term Memory (Conversation Summary)

* Maintains a **short episodic summary** of the conversation.
* Stored in `st.session_state.long_term_summary`.
* Purpose: Retain important details over multiple chat turns without overloading short-term memory.
* **Update mechanism**:

  1. Every `MEMORY_WINDOW` messages, `update_long_term_summary()` is called.
  2. It sends **recent messages + previous summary** to Groq LLM.
  3. LLM returns a **concise 3–4 line summary**.
  4. This summary is saved back to `st.session_state.long_term_summary`.
* Example format:

```
User has minor cuts on hand. User is anxious about bleeding. Buddy advised cleaning and bandaging. User followed steps.
```

---

## 3️⃣ Combining Memories for Context

* When processing a user query, the chatbot combines:

  1. **Long-term summary** (`st.session_state.long_term_summary`) → keeps track of history.
  2. **Recent short-term messages** (`chat_history[-MEMORY_WINDOW:]`) → maintains immediate context.
* This combined context is fed into the **RAG chain** to generate a response.

---

## 4️⃣ Deep Think Mode

* If **Deep Think** is ON:

  * Gemini first **summarizes the intent** of the user’s query.
  * Groq receives **both the Gemini summary + raw query** for context-aware answers.
* If **Deep Think** is OFF:

  * Groq uses **raw user query + combined memory** without Gemini summarization.
* Purpose: Give either **detailed, thoughtful answers** or **fast, direct answers**.

---

## 5️⃣ Key Variables

| Variable             | Purpose                                                    |
| -------------------- | ---------------------------------------------------------- |
| `chat_history`       | Stores recent human & AI messages.                         |
| `long_term_summary`  | Stores episodic memory summary.                            |
| `message_counter`    | Counts total messages to determine when to update summary. |
| `MEMORY_WINDOW`      | Number of messages in short-term context.                  |
| `deep_think_enabled` | Toggle to use Gemini summarization.                        |

---

## 6️⃣ Summary Flow

```
User Message → Append to chat_history
               ↓
Combine chat_history[-MEMORY_WINDOW:] + long_term_summary
               ↓
(Optional) Gemini summarizes intent → final_input
               ↓
Groq RAG chain → AI response
               ↓
Append AI response to chat_history
               ↓
If message_counter % MEMORY_WINDOW == 0 → update_long_term_summary()
```

* This ensures the chatbot **remembers important details**, **keeps answers context-aware**, and **avoids repeating info unnecessarily**.

---

Made with ❤️‍🩹 to make your AI feel like a **real buddy with memory**.

---

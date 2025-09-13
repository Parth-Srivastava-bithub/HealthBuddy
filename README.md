
# ❤️‍🩹 AI-Guided First-Aid Support Chatbot

This is a **Streamlit-based First-Aid Assistant** that combines dual LLMs (**Groq** + **Gemini**) with local embeddings and FAISS for **RAG (Retrieval-Augmented Generation)**.  
It helps users with **basic first-aid** and **stress support** in a **casual, buddy-like tone** – not like a boring medical book! 🚑

---

## ✨ Features

- 🤖 **Dual LLM pipeline**:
  - **Gemini** → Intent summarizer ("Deep Think" mode).
  - **Groq** → Final RAG-based answers.
- 📚 **Local FAISS vector store** for first-aid docs.
- 🧠 **Conversational memory**:
  - Short-term (windowed history).
  - Long-term (periodic summarization).
- 🎮 **Gamification**:
  - Earn points for on-topic queries or quick-access buttons.
  - Level up: *Beginner → Learner → Practitioner → Guardian → Expert*.
- 🎛️ **Deep Think toggle**:
  - ON → Gemini analyzes intent for more contextual replies.
  - OFF → Fast, direct answers from Groq.
- 🌑 **Custom Dark UI Theme** (GitHub dark vibes).
- ⚡ **Quick-Access Buttons** for:
  - Minor bleeding 🩸  
  - Minor burn 🔥  
  - Anxiety attack 😟  

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLMs**: 
  - [Groq](https://groq.com/) – `gemma2-9b-it`
  - [Google Gemini](https://ai.google.dev/) – `gemini-1.5-flash-latest`
- **Embeddings**: [HuggingFace](https://huggingface.co/) – `all-MiniLM-L6-v2`
- **Vector DB**: FAISS (persistent local index)
- **Environment Vars**: `.env` for API keys

---

## 📂 Project Structure

```

📁 ai-first-aid-chatbot
│── app.py              # Main Streamlit app
│── requirements.txt    # Python dependencies
│── .env.example        # Sample env file
│── faiss\_first\_aid\_index/   # FAISS index storage
│── README.md           # This file

````

---

## ⚡ Setup & Run

### 1. Clone repo
```bash
git clone https://github.com/your-username/ai-first-aid-chatbot.git
cd ai-first-aid-chatbot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run app

```bash
streamlit run app.py
```

---

## 🎮 Gamification Rules

* **On-topic query** → +15 points
* **Minor bleeding button** → +10 points
* **Minor burn button** → +10 points
* **Anxiety button** → +20 points
* Daily cap: **100 points**
* Button cooldown: **5 min**

---

## ⚠️ Disclaimer

This chatbot is for **educational and informational purposes only**.
It is **NOT a replacement for professional medical advice**. For emergencies, **always call a doctor or emergency services**.

---

## 🚀 Future Improvements

* 🔊 Voice input/output support
* 📱 Mobile-first UI
* 🌍 Multi-language support
* 🧩 Plugin system for adding more health modules

---

Made with ❤️‍🩹 and ☕ by Parth Srivastava

```


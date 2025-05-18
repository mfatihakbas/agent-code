# 🤖 ML Question Answering Assistant

A smart AI-powered assistant that answers machine learning-related questions.  
It first searches your Airtable-based knowledge base using embedding similarity.  
If no close match is found, it queries OpenAI (GPT-3.5 or GPT-4) and stores the new answer — learning over time.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agent-code.git
cd agent-code/agent
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root with the following content:

```env
OPENAI_API_KEY=your_openai_api_key
AIRTABLE_API_KEY=your_airtable_api_key
AIRTABLE_BASE_ID=your_airtable_base_id
AIRTABLE_TABLE_NAME=Table 1
```

> You can create a `.env.example` for others to follow.

---

## 🧠 How It Works

### Start the assistant

```bash
python main.py
```

You’ll be prompted to ask questions:

```
❓ Enter your question: What is overfitting?
```

- The app first checks your Airtable for similar past questions (via embedding match).
- If found, it returns the saved answer and updates the usage count.
- If not found, it queries OpenAI (GPT) and saves the new Q&A in Airtable.

---

## 📊 Features

- ✅ Embedding-based similarity search using SentenceTransformers
- ✅ Vector comparison with cosine similarity
- ✅ Airtable as a persistent and extensible vector store
- ✅ Tracks usage count of each record
- ✅ Falls back to OpenAI GPT-3.5/GPT-4 when no match is found
- ✅ Response source and search duration are printed
- ✅ Initial sample questions are added automatically on first run

---

## 📁 Project Structure

```
agent-code/
│
├── agent/
│   ├── main.py                # Main application entry
│   ├── airtable_client.py     # Embedding + Airtable logic
│   ├── openai_client.py       # OpenAI API integration
│   ├── utils.py               # Helper utilities (e.g., cleaning, filtering)
│   └── requirements.txt       # Dependencies list
│
├── .env                       # API keys (not committed)
└── README.md
```

---

## 🛠️ Key Dependencies

This project relies on the following libraries:

- `openai`
- `pyairtable`
- `sentence-transformers`
- `torch`
- `transformers`
- `scikit-learn`
- `numpy`
- `python-dotenv`
- `requests`

All are listed in `requirements.txt`.

---

## 🤝 Contributing

Pull requests and ideas are welcome! Please open an issue before making major changes.

---

## 🛡️ License

MIT License © 2025

# 📡 Telecom Data Anomaly Agent

A LangChain-powered conversational AI assistant for analyzing Telecom KPI data. This agent combines NVIDIA LLMs, LangGraph workflows, and internal analytics tools to detect anomalies, summarize trends, and explain KPI behavior across cellular network sectors.

---

## 🚀 Features

- 📊 **KPI Analysis**: Understand SINR, Throughput, Call Drop Rate, and more
- ⚠️ **Anomaly Detection**: Uses DWT-MLEAD, Isolation Forest & ensemble voting
- 🤖 **Conversational Interface**: Chat-style UI using Gradio
- 🔍 **Search-Augmented Reasoning**: Integrates Tavily Search API
- 📁 **Tool-Driven Agent**: Modular LangChain tools for in-depth queries

---

## 🧠 Example Queries

- “Were there anomalies in DL_Throughput last week?”
- “Which day did SINR drop significantly at SITE_007?”
- “Show the day with the highest packet loss.”
- “Which KPIs were anomalous together on 2024-06-12?”

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Telecom_data_anomaly_Agent.git
cd Telecom_data_anomaly_Agent
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 🔐 API Keys Configuration
This project uses:
NVIDIA API for LLM access

Tavily API for web search

You can provide them in either of two ways:

Option A: Hardcode in Agent.py
```bash
nvidia_key = "your-nvidia-api-key"
tavily_key = "your-tavily-api-key"
```
Option B: Use a .env File
Create a .env file in the root directory:
```bash
NVIDIA_API_KEY=your-nvidia-api-key
TAVILY_API_KEY=your-tavily-api-key
```

## ⚙️ Running the App
Step 1: Launch the LangChain Agent Server
```bash
python MCP_server.py
```
Step 2: Launch the Gradio Chat UI
In a new terminal:
```bash
python app.py
```

Open http://localhost:7860 to chat with the assistant!

## 📂 Folder Structure
```
Telecom_data_anomaly_Agent/
│
├── Agent.py            # LangChain agent with tool bindings
├── MCP_server.py       # FastAPI server for backend
├── app.py              # Gradio UI
├── tools.py            # Custom tools: anomaly analysis, KPI summaries
├── requirements.txt
├── .gitignore
├── Data/
│   ├── df_ensemble.csv
│   └── KPI_data_cleaned.csv
├── README.md
```

## 🧪 Models Used
Model	Purpose
DWT-MLEAD	Shape-based anomaly detection in time series
IsolationForest	Tree-based unsupervised outlier detection
Ensemble Voting	Combines above for robustness

## 📖 Reference: Schmidl et al., Anomaly Detection in Time Series (2022)

## 📋 Evaluation Approach
Visual inspection of anomaly overlays

Anomaly co-occurrence across related KPIs

Anomaly rate expected between 0.5–5%

## 🧠 Credits
Built using LangChain, Gradio, FastAPI

Powered by NVIDIA LLMs

## 📜 License
MIT License – feel free to use and modify.

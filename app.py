import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #1e293b;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
    }

    /* Weather */
    .weather-card {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.25);
        margin-top: 20px;
    }

    .temperature {
        font-size: 48px;
        font-weight: 800;
    }

    .weather-text {
        font-size: 18px;
        opacity: 0.95;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------

search_tool = TavilySearch(max_results=3)


@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not configured."

    try:
        url = (
            "http://api.weatherstack.com/current"
            f"?access_key={api_key}"
            f"&query={city}"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather for {city}."

        temperature = data["current"]["temperature"]
        condition = data["current"]["weather_descriptions"][0]
        humidity = data["current"]["humidity"]

        return (
            f"City: {city}\n"
            f"Temperature: {temperature}°C\n"
            f"Weather: {condition}\n"
            f"Humidity: {humidity}%"
        )

    except Exception as e:
        return f"Weather service error: {str(e)}"


# ---------------------------------------------------------
# CREATE AI AGENT
# ---------------------------------------------------------

@st.cache_resource
def create_ai_agent():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY
    )

    tools = [
        search_tool,
        get_weather
    ]

    agent = create_agent(
        model=llm,
        tools=tools
    )

    return agent


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 🤖 AI Agent")

    st.markdown("---")

    st.markdown("### 🛠️ Available Tools")

    st.success("🔎 Tavily Web Search")

    st.success("🌤️ WeatherStack")

    st.success("🧠 Groq LLM")

    st.markdown("---")

    st.markdown("### 💡 Try asking")

    st.markdown("""
    **Weather**
    
    `What is the current weather in Ongole?`

    **Web Search**
    
    `What are the latest AI developments?`

    **General**
    
    `Explain artificial intelligence simply.`
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Agent Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent assistant with Web Search & Weather tools'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# API KEY CHECK
# ---------------------------------------------------------

missing_keys = []

if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not WEATHERSTACK_API_KEY:
    missing_keys.append("WEATHERSTACK_API_KEY")

if missing_keys:

    st.warning(
        "⚠️ Missing API keys: "
        + ", ".join(missing_keys)
        + ". Please add them to your `.env` file."
    )

    st.stop()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# WELCOME CARD
# ---------------------------------------------------------

if len(st.session_state.messages) == 0:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            👋 Welcome!
        </div>

        <p>
            I am an AI agent capable of answering questions,
            searching the web, and checking current weather.
        </p>

        <p>
            <b>Ask me anything to get started.</b>
        </p>

    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="🤖" if message["role"] == "assistant" else "👤"
    ):
        st.markdown(message["content"])


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

user_input = st.chat_input(
    "Ask your AI agent anything..."
)


if user_input:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("🤔 Thinking..."):

            try:

                agent = create_ai_agent()

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                answer = response["messages"][-1].content

                st.markdown(answer)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                answer = (
                    "❌ Something went wrong.\n\n"
                    f"**Error:** `{str(e)}`"
                )

                st.error(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
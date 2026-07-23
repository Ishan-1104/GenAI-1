"""
CityPulse AI 🌆 — An agentic city-intelligence chatbot with human-in-the-loop
tool approval, built with LangChain + Mistral + Streamlit.

Run locally:
    streamlit run app.py
"""

import os
import re
import html
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityPulse AI",
    page_icon="🌆",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# STYLE
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at top left, #101425 0%, #0b0e17 55%, #05060a 100%);
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 20px;
            background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
            box-shadow: 0 12px 40px rgba(124, 58, 237, 0.35);
            margin-bottom: 1.4rem;
        }
        .hero h1 {
            color: white;
            font-size: 2rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255,255,255,0.88);
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }
        .hero .badges {margin-top: 0.7rem;}
        .pill {
            display: inline-block;
            padding: 0.18rem 0.65rem;
            margin-right: 0.4rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            color: white;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .tool-card {
            border: 1px solid rgba(124, 58, 237, 0.35);
            background: rgba(124, 58, 237, 0.08);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.88rem;
        }
        .tool-card .tool-title {
            font-weight: 700;
            color: #c4b5fd;
            margin-bottom: 0.15rem;
        }
        .result-card {
            border: 1px solid rgba(16, 185, 129, 0.35);
            background: rgba(16, 185, 129, 0.07);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0 0.9rem 0;
            font-size: 0.88rem;
            white-space: pre-wrap;
        }
        .result-card .result-title {
            font-weight: 700;
            color: #6ee7b7;
            margin-bottom: 0.2rem;
        }
        .denied-card {
            border: 1px solid rgba(239, 68, 68, 0.4);
            background: rgba(239, 68, 68, 0.08);
            border-radius: 14px;
            padding: 0.6rem 1rem;
            margin: 0.4rem 0 0.9rem 0;
            font-size: 0.85rem;
            color: #fca5a5;
            font-weight: 600;
        }
        .approval-box {
            border: 1px solid rgba(251, 191, 36, 0.45);
            background: rgba(251, 191, 36, 0.08);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin: 0.6rem 0 1rem 0;
        }
        .approval-box h4 {
            margin: 0 0 0.5rem 0;
            color: #fcd34d;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] {
            background: #0c0f1a;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🌆 CityPulse AI</h1>
        <p>An agentic assistant that checks live weather &amp; news for any city —
        with human-in-the-loop approval before it acts.</p>
        <div class="badges">
            <span class="pill">LangChain</span>
            <span class="pill">Mistral LLM</span>
            <span class="pill">Tool Calling</span>
            <span class="pill">Human-in-the-loop</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIG
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mistral_key = st.text_input(
        "Mistral API Key", type="password", value=os.getenv("MISTRAL_API_KEY", "")
    )
    weather_key = st.text_input(
        "OpenWeather API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", "")
    )
    tavily_key = st.text_input(
        "Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", "")
    )

    model_name = st.selectbox(
        "Model",
        ["mistral-small-2506", "mistral-large-latest", "open-mistral-nemo"],
        index=0,
    )

    require_approval = st.toggle("🔐 Require approval before tool calls", value=True)

    st.divider()
    st.markdown("### 📊 Session stats")
    c1, c2 = st.columns(2)
    c1.metric("Messages", st.session_state.get("msg_count", 0))
    c2.metric("Tool calls", st.session_state.get("tool_count", 0))

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        for key in ["chat_history", "lc_messages", "pending", "msg_count", "tool_count"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption("Built with ❤️ using Streamlit, LangChain & Mistral AI")

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # for rendering
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [
        SystemMessage(
            content=(
                "You are CityPulse, a friendly and concise city-intelligence assistant. "
                "Use your tools to fetch live weather and news when the user asks about a "
                "city. Always mention the city name in your final answer."
            )
        )
    ]
if "pending" not in st.session_state:
    st.session_state.pending = None
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0
if "tool_count" not in st.session_state:
    st.session_state.tool_count = 0


# ────────────────────────────────────────────────────────────────────────────
# TOOLS
# ────────────────────────────────────────────────────────────────────────────
def build_tools(weather_api_key: str, tavily_api_key: str):
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a given city."""
        if not weather_api_key:
            return "Error: OpenWeather API key is not configured."
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={weather_api_key}&units=metric"
        )
        try:
            data = requests.get(url, timeout=10).json()
        except Exception as e:
            return f"Error fetching weather: {e}"

        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather for ' + city)}"

        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        return (
            f"Weather in {city}: {desc}, {temp}°C (feels like {feels}°C), "
            f"humidity {humidity}%, wind {wind} m/s"
        )

    @tool
    def get_news(city: str) -> str:
        """Get the latest news headlines about a given city."""
        if not tavily_api_key:
            return "Error: Tavily API key is not configured."
        try:
            client = TavilyClient(api_key=tavily_api_key)
            response = client.search(
                query=f"latest news in {city}", search_depth="basic", max_results=3
            )
        except Exception as e:
            return f"Error fetching news: {e}"

        results = response.get("results", [])
        if not results:
            return f"No news found for {city}"

        lines = []
        for r in results:
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            lines.append(f"- {title} ({url}): {content}")
        return f"Latest news in {city}:\n" + "\n".join(lines)

    return [get_weather, get_news]


# ────────────────────────────────────────────────────────────────────────────
# AGENT LOOP HELPERS
# ────────────────────────────────────────────────────────────────────────────
def get_llm_with_tools(tools):
    llm = ChatMistralAI(model_name=model_name, api_key=mistral_key)
    return llm.bind_tools(tools)


def execute_tool_calls(ai_message, tools_by_name, approve: bool):
    """Execute (or deny) every tool call attached to ai_message, updating state."""
    for tc in ai_message.tool_calls:
        tool_name = tc["name"]
        st.session_state.chat_history.append(
            {"role": "tool_call", "name": tool_name, "args": tc["args"]}
        )
        st.session_state.tool_count += 1

        if approve:
            tool_fn = tools_by_name[tool_name]
            try:
                output = tool_fn.invoke(tc["args"])
            except Exception as e:
                output = f"Error running {tool_name}: {e}"
            st.session_state.lc_messages.append(
                ToolMessage(content=str(output), tool_call_id=tc["id"])
            )
            st.session_state.chat_history.append(
                {"role": "tool_result", "name": tool_name, "content": str(output)}
            )
        else:
            st.session_state.lc_messages.append(
                ToolMessage(content="Tool call denied by the user.", tool_call_id=tc["id"])
            )
            st.session_state.chat_history.append({"role": "tool_denied", "name": tool_name})


def run_agent_loop(llm_with_tools, tools_by_name):
    """Keep stepping the agent until it produces a final answer or needs approval."""
    while True:
        result = llm_with_tools.invoke(st.session_state.lc_messages)
        st.session_state.lc_messages.append(result)

        if result.tool_calls:
            if require_approval:
                st.session_state.pending = {"ai_message": result}
                return
            execute_tool_calls(result, tools_by_name, approve=True)
            continue
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.content}
            )
            return


# ────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ────────────────────────────────────────────────────────────────────────────
URL_PATTERN = re.compile(r"(https?://[^\s\)]+)")


def linkify(text: str) -> str:
    """HTML-escape text, then turn raw URLs into clickable <a> tags and newlines into <br>."""
    escaped = html.escape(text)

    def _wrap(match):
        url = match.group(1)
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#6ee7b7;text-decoration:underline;">{url}</a>'
        )

    linked = URL_PATTERN.sub(_wrap, escaped)
    return linked.replace("\n", "<br>")


# ────────────────────────────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ────────────────────────────────────────────────────────────────────────────
def render_history():
    for item in st.session_state.chat_history:
        role = item["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(item["content"])
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🌆"):
                st.markdown(item["content"])
        elif role == "tool_call":
            icon = "☀️" if item["name"] == "get_weather" else "📰"
            args_str = ", ".join(f"{k}={v!r}" for k, v in item["args"].items())
            st.markdown(
                f"""<div class="tool-card">
                        <div class="tool-title">{icon} Calling <code>{item['name']}</code></div>
                        {args_str}
                    </div>""",
                unsafe_allow_html=True,
            )
        elif role == "tool_result":
            icon = "☀️" if item["name"] == "get_weather" else "📰"
            content = item["content"].replace("\n", "<br>")
            st.markdown(
                f"""<div class="result-card">
                        <div class="result-title">{icon} Result</div>
                        {content}
                    </div>""",
                unsafe_allow_html=True,
            )
        elif role == "tool_denied":
            st.markdown(
                f"""<div class="denied-card">🚫 {item['name']} was denied by you.</div>""",
                unsafe_allow_html=True,
            )


render_history()

# ────────────────────────────────────────────────────────────────────────────
# PENDING APPROVAL UI
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.pending is not None:
    ai_message = st.session_state.pending["ai_message"]
    tools = build_tools(weather_key, tavily_key)
    tools_by_name = {t.name: t for t in tools}

    with st.container():
        st.markdown('<div class="approval-box">', unsafe_allow_html=True)
        st.markdown("#### 🔐 The agent wants to use a tool")
        for tc in ai_message.tool_calls:
            icon = "☀️" if tc["name"] == "get_weather" else "📰"
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            st.markdown(f"{icon} **`{tc['name']}`**  —  {args_str}")

        col1, col2 = st.columns(2)
        approve_clicked = col1.button("✅ Approve", use_container_width=True)
        deny_clicked = col2.button("🚫 Deny", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if approve_clicked or deny_clicked:
        execute_tool_calls(ai_message, tools_by_name, approve=approve_clicked)
        st.session_state.pending = None
        llm_with_tools = get_llm_with_tools(tools)
        run_agent_loop(llm_with_tools, tools_by_name)
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# QUICK PROMPTS (only shown on empty chat)
# ────────────────────────────────────────────────────────────────────────────
elif not st.session_state.chat_history:
    st.markdown("##### 💡 Try one of these")
    qc1, qc2, qc3 = st.columns(3)
    quick_prompt = None
    if qc1.button("☀️ Weather in Mumbai", use_container_width=True):
        quick_prompt = "What's the weather in Mumbai right now?"
    if qc2.button("📰 News in Bangalore", use_container_width=True):
        quick_prompt = "Give me the latest news in Bangalore."
    if qc3.button("🌆 Full report on Delhi", use_container_width=True):
        quick_prompt = "Give me a quick weather and news report for Delhi."

    if quick_prompt:
        st.session_state["_queued_prompt"] = quick_prompt
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.pending is None:
    keys_ready = bool(mistral_key)
    user_input = st.chat_input(
        "Ask about a city's weather or news..." if keys_ready else "Add your Mistral API key in the sidebar first"
    )

    queued = st.session_state.pop("_queued_prompt", None)
    final_input = user_input or queued

    if final_input:
        if not mistral_key:
            st.warning("⚠️ Please add your Mistral API key in the sidebar to chat.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": final_input})
            st.session_state.lc_messages.append(HumanMessage(final_input))
            st.session_state.msg_count += 1

            tools = build_tools(weather_key, tavily_key)
            tools_by_name = {t.name: t for t in tools}
            llm_with_tools = get_llm_with_tools(tools)

            with st.spinner("CityPulse is thinking..."):
                run_agent_loop(llm_with_tools, tools_by_name)

            st.rerun()
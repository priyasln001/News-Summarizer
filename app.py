import streamlit as st
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or DEFAULT_NEWSAPI_KEY

try:
    import openai
    from openai.error import OpenAIError, RateLimitError, AuthenticationError
except ImportError as exc:
    raise ImportError("OpenAI is required to run this app. Install with 'pip install openai'.") from exc

openai.api_key = OPENAI_API_KEY or None

st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="wide")

st.title("📰 AI News Summarizer")
st.caption("Enter a topic and get the latest headlines with AI-generated bullet summaries.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    country = st.selectbox("Country", ["in", "us", "gb", "ca"], index=1)
    category = st.selectbox(
        "Category",
        ["general", "business", "technology", "sports", "health", "entertainment"],
    )

# Main inputs
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("🔍 Search Topic", placeholder="AI, Cricket, Stock Market, Politics")
with col2:
    num_articles = st.slider("No. of Articles", min_value=3, max_value=10, value=5)

if not NEWSAPI_KEY:
    st.warning("NewsAPI key is missing. Please add it to DEFAULT_NEWSAPI_KEY at the top of app.py.")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key is missing. Please add it to DEFAULT_OPENAI_API_KEY at the top of app.py.")

button_disabled = not (OPENAI_API_KEY and NEWSAPI_KEY)


def format_published_at(iso_string: str) -> str:
    if not iso_string:
        return "Unknown date"
    try:
        return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_string


def fetch_news(topic: str, page_size: int, api_key: str) -> list:
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": topic,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(data.get("message", "NewsAPI returned an error."))
    return data.get("articles", [])


def summarize_article(title: str, description: str) -> str:
    text = description or title or "No content available."
    prompt = (
        "Summarize the following news article in 3 concise bullet points. "
        "Be factual and concise, and use only information from the article.\n\n"
        f"Title: {title}\n"
        f"Content: {text}\n\n"
        "Return exactly 3 bullet points, each beginning with '- '."
    )
    # extractive fallback: pick up to 3 short sentences from the description/title
    def extractive_summary(src: str, max_sentences: int = 3) -> str:
        if not src:
            return "No summary available."
        # naive sentence split
        parts = [s.strip() for s in src.replace('\n', ' ').split('. ') if s.strip()]
        selected = parts[:max_sentences]
        bullets = '\n'.join(f"- {s.rstrip('.')}" for s in selected)
        return bullets or "No summary available."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise news summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message["content"].strip()
    except RateLimitError:
        st.warning("OpenAI quota exceeded — using local fallback summary.")
        return extractive_summary(text, 3)
    except AuthenticationError:
        st.error("OpenAI authentication failed. Using local fallback summary.")
        return extractive_summary(text, 3)
    except OpenAIError:
        st.warning("OpenAI request failed — using local fallback summary.")
        return extractive_summary(text, 3)


def summarize_overall(topic: str, articles: list) -> str:
    article_text = "\n\n".join(
        f"Title: {a.get('title', 'No title')}\nDescription: {a.get('description', 'No description')}"
        for a in articles
    )

    prompt = (
        "Based on these news headlines and descriptions, provide a single paragraph overall summary "
        f"for the topic '{topic}'. Keep it factual, concise, and avoid speculation.\n\n{article_text}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise news summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message["content"].strip()
    except (RateLimitError, OpenAIError, AuthenticationError):
        # fallback: build a short paragraph from titles/descriptions
        heads = []
        for a in articles[:5]:
            t = a.get('title') or ''
            d = a.get('description') or ''
            heads.append(t if d == '' else f"{t}: {d}")
        short = ' '.join(heads)
        if not short:
            return "No overall summary available."
        # take first 2 sentences
        sents = [s.strip() for s in short.replace('\n', ' ').split('. ') if s.strip()]
        para = '. '.join(sents[:2])
        if not para.endswith('.'):
            para = para + '.'
        return para


if st.button("Get News + Summary 🚀", disabled=button_disabled):
    if not topic:
        st.error("Please enter a topic to search for news.")
    elif not OPENAI_API_KEY or not NEWSAPI_KEY:
        st.error("API keys are required to fetch news and generate summaries.")
    else:
        with st.spinner("Fetching latest news..."):
            try:
                articles = fetch_news(topic, num_articles, NEWSAPI_KEY)
            except Exception as exc:
                st.error(f"Unable to fetch news: {exc}")
                st.stop()

            if not articles:
                st.warning("No articles were found for that query.")
                st.stop()

            st.success(f"Found {len(articles)} articles")

            for index, article in enumerate(articles, start=1):
                title = article.get("title", "No title")
                published = format_published_at(article.get("publishedAt", ""))
                source = article.get("source", {}).get("name", "Unknown source")
                description = article.get("description") or article.get("content") or "No description available."
                url = article.get("url") or ""
                image_url = article.get("urlToImage")

                with st.container():
                    header_col, image_col = st.columns([3, 1])
                    with header_col:
                        st.subheader(f"{index}. {title}")
                        st.caption(f"Source: {source} | Published: {published}")
                        summary = summarize_article(title, description)
                        st.markdown("**AI Summary:**")
                        st.markdown(summary)
                        if url:
                            st.markdown(f"[Read Full Article →]({url})")

                    with image_col:
                        if image_url:
                            st.image(image_url, use_column_width=True)

            st.divider()
            st.subheader("📊 Overall Topic Summary")
            overall_summary = summarize_overall(topic, articles)
            st.write(overall_summary)

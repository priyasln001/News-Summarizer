import streamlit as st
import os
import requests
import re
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError

# Load environment variables
load_dotenv()

# API Keys (Local + Streamlit Cloud)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY") or st.secrets.get("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")

# OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Streamlit Page Config
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .article-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .article-card:hover {
        border-color: #4F46E5;
        transform: translateY(-2px);
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 12px;
        background: rgba(79, 70, 229, 0.15);
        color: #818CF8;
        margin-right: 8px;
    }
    .summary-box {
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #06B6D4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📰 AI News Summarizer</h1>', unsafe_allow_html=True)
st.caption("Discover top headlines and search global news with AI-powered summaries.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Fetch Mode",
        ["Topic Search", "Top Headlines"],
        index=0
    )

    country = st.selectbox(
        "Country (Top Headlines)",
        ["us", "in", "gb", "ca", "au", "de"],
        index=1
    )

    category = st.selectbox(
        "Category (Top Headlines)",
        ["general", "business", "technology", "sports", "health", "entertainment", "science"],
        index=0
    )

    sort_by = st.selectbox(
        "Sort By",
        ["relevancy", "publishedAt", "popularity"],
        index=0
    )

    num_articles = st.slider(
        "No. of Articles",
        min_value=3,
        max_value=10,
        value=5
    )

# Main Inputs
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "🔍 Search Topic",
        placeholder="AI, Stock Market, Cricket, Space, Tech",
        value="" if mode == "Top Headlines" else "Technology"
    )

# Status notifications
if not NEWSAPI_KEY:
    st.error("⚠️ `NEWSAPI_KEY` is missing. Please add it to your `.env` or Streamlit secrets.")

if not OPENAI_API_KEY:
    st.info("ℹ️ `OPENAI_API_KEY` is missing or out of quota. Using Smart Local NLP Summarizer.")

button_disabled = not NEWSAPI_KEY


def format_published_at(iso_string: str) -> str:
    if not iso_string:
        return "Unknown date"
    try:
        return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).strftime("%b %d, %Y • %H:%M")
    except ValueError:
        return iso_string


def smart_extractive_summary(title: str, description: str, content: str = "") -> str:
    """Frequency-based extractive NLP summarizer (works without OpenAI API)."""
    full_text = f"{title}. {description or ''}. {content or ''}".strip()
    # Clean text
    clean_text = re.sub(r'\[\+\d+ chars\]', '', full_text)
    words = re.findall(r'\w+', clean_text.lower())
    
    stopwords = set(["the", "a", "an", "in", "on", "of", "and", "is", "to", "for", "with", "that", "this", "by", "from", "at", "as", "be", "are", "it", "has", "have", "was", "were", "or", "an", "not", "but", "more", "says", "said"])
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not filtered_words:
        return f"- **{title}**\n- {description or 'No further details available.'}"

    freq = Counter(filtered_words)
    
    # Score sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', clean_text) if len(s.strip()) > 15]
    if not sentences:
        return f"- **{title}**"
    
    sentence_scores = {}
    for i, s in enumerate(sentences):
        score = sum(freq[w.lower()] for w in re.findall(r'\w+', s) if w.lower() in freq)
        sentence_scores[i] = score / (len(s.split()) + 1)
        
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    top_indices.sort()
    
    selected = [sentences[idx].rstrip('.') for idx in top_indices]
    bullets = "\n".join(f"- {s}." for s in selected)
    return bullets if bullets else f"- {title}"


def fetch_news(topic: str, page_size: int, api_key: str, mode: str, country: str, category: str, sort_by: str) -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    if mode == "Top Headlines":
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "country": country,
            "category": category,
            "pageSize": page_size,
        }
        if topic.strip():
            params["q"] = topic
    else:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": topic if topic.strip() else category,
            "language": "en",
            "sortBy": sort_by,
            "pageSize": page_size,
        }

    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()

    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(data.get("message", "NewsAPI Error"))

    return data.get("articles", [])


def summarize_article(title: str, description: str, content: str = "") -> str:
    if not client:
        return smart_extractive_summary(title, description, content)

    text = description or title or "No content available."
    prompt = f"""
Summarize the following news article in exactly 3 concise, impactful bullet points.

Title: {title}
Content: {text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise news summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()

    except (RateLimitError, AuthenticationError, OpenAIError):
        return smart_extractive_summary(title, description, content)


def summarize_overall(topic_name: str, articles: list) -> str:
    article_text = "\n\n".join(
        f"Title: {a.get('title', 'No title')}\nDescription: {a.get('description', 'No description')}"
        for a in articles
    )

    if client:
        prompt = f"""
Based on these news articles, write one clear, cohesive executive summary paragraph about '{topic_name}'.

Articles:
{article_text}
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an executive news analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=250,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, AuthenticationError, OpenAIError):
            pass

    # High-quality fallback overview
    bullets = []
    for a in articles[:5]:
        t = a.get("title")
        d = a.get("description")
        if t and d:
            bullets.append(f"• **{t}**: {d}")
        elif t:
            bullets.append(f"• **{t}**")

    if not bullets:
        return "No overall summary available."

    return f"### Executive Overview for '{topic_name}'\n\n" + "\n\n".join(bullets)


# Fetch Trigger
if st.button("Get News + Summary 🚀", disabled=button_disabled, type="primary"):
    
    search_query = topic.strip() if topic.strip() else category
    with st.spinner("Fetching latest news..."):
        try:
            articles = fetch_news(
                search_query,
                num_articles,
                NEWSAPI_KEY,
                mode,
                country,
                category,
                sort_by
            )
        except Exception as e:
            st.error(f"Unable to fetch news: {e}")
            st.stop()

    if not articles:
        st.warning("No news articles found for this topic/filter. Try another search.")
        st.stop()

    st.success(f"Fetched {len(articles)} latest articles")
    
    summary_report_markdown = f"# News Summary Report: {search_query.title()}\n\n"

    for i, article in enumerate(articles, start=1):
        title = article.get("title", "No Title")
        description = article.get("description") or ""
        content = article.get("content") or ""
        source = article.get("source", {}).get("name", "Unknown Source")
        published = format_published_at(article.get("publishedAt", ""))
        url = article.get("url", "")
        image = article.get("urlToImage")

        summary = summarize_article(title, description, content)

        summary_report_markdown += f"## {i}. {title}\n*Source: {source} | Date: {published}*\n\n{summary}\n\n[Read Article]({url})\n\n---\n\n"

        # Article Render
        left, right = st.columns([3, 1])
        with left:
            st.subheader(f"{i}. {title}")
            st.markdown(f'<span class="badge">{source}</span> <span style="color:#9CA3AF; font-size:0.85rem;">📅 {published}</span>', unsafe_allow_html=True)
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.markdown("**AI Summary**")
            st.markdown(summary)
            st.markdown('</div>', unsafe_allow_html=True)
            if url:
                st.markdown(f"[🔗 Read Full Article]({url})")

        with right:
            if image:
                try:
                    st.image(image, width="stretch")
                except Exception:
                    st.image(image, use_container_width=True)

        st.divider()

    st.subheader("📊 Overall Topic Summary")
    overall = summarize_overall(search_query, articles)
    st.markdown(overall)
    summary_report_markdown += f"## Overall Summary\n\n{overall}\n"

    # Export Download Button
    st.download_button(
        label="📥 Download News Summary Report (.md)",
        data=summary_report_markdown,
        file_name=f"News_Summary_{search_query.replace(' ', '_')}.md",
        mime="text/markdown"
    )
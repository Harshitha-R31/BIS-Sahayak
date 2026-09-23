import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BIS Sahayak",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

BASE = Path(__file__).parent

with open(
    BASE / "knowledge_base.json",
    "r",
    encoding="utf-8"
) as file:
    DOCS = json.load(file)


# =========================================================
# SEARCH ENGINE
# =========================================================

corpus = []

for doc in DOCS:
    text = (
        doc.get("title", "")
        + " "
        + doc.get("content", "")
        + " "
        + " ".join(doc.get("keywords", []))
    )

    corpus.append(text)


vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

matrix = vectorizer.fit_transform(corpus)


def retrieve(query, number=3):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    indexes = scores.argsort()[::-1][:number]

    results = []

    for index in indexes:

        if scores[index] > 0:

            results.append(
                (
                    DOCS[index],
                    float(scores[index])
                )
            )

    return results


# =========================================================
# ANSWER GENERATOR
# =========================================================

def generate_answer(query, results):

    if not results:

        return (
            "I could not find enough information in the current "
            "BIS knowledge base. Please try a more specific "
            "question about standards, certification, "
            "hallmarking or testing."
        )

    q = query.lower()

    # Certification questions
    if any(
        word in q
        for word in [
            "certification",
            "certificate",
            "certify",
            "licence",
            "license"
        ]
    ):

        return (
            "For BIS product certification, the first step is "
            "to identify the Indian Standard applicable to the "
            "product. The manufacturer then follows the relevant "
            "BIS conformity-assessment requirements, including "
            "applicable manufacturing, quality-control and "
            "testing requirements."
        )

    # Hallmarking questions
    if any(
        word in q
        for word in [
            "hallmark",
            "hallmarking",
            "huid",
            "gold",
            "silver",
            "jewellery",
            "jewelry"
        ]
    ):

        return (
            "BIS provides hallmarking resources for consumers "
            "and industry. HUID is a unique alphanumeric "
            "identification associated with a hallmarked item "
            "and can be used for verification through BIS "
            "consumer services."
        )

    # Testing questions
    if any(
        word in q
        for word in [
            "lab",
            "laboratory",
            "laboratories",
            "testing",
            "test centre",
            "test center"
        ]
    ):

        return (
            "BIS provides laboratory and testing-related "
            "services to support conformity assessment. "
            "The official BIS laboratory services resource "
            "provides information about laboratory-related "
            "services and testing support."
        )

    # Default standards answer
    return (
        "For standards discovery, BIS provides resources such "
        "as Know Your Standard and the BIS Standards Portal. "
        "Users can search using an IS number, keyword or "
        "product name to find relevant standards-related "
        "information."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("BIS Sahayak")

    st.caption(
        "Find • Understand • Comply"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📄 Standards",
            "✓ Certification",
            "◇ Hallmarking",
            "⚗ Testing & Labs",
            "ⓘ About BIS"
        ]
    )

    st.divider()

    st.info(
        "Evidence-first assistant\n\n"
        "Answers are grounded in the curated BIS "
        "knowledge base and include official BIS "
        "source links."
    )

    st.caption(
        "ASK → RETRIEVE → RANK → GROUND → CITE → ANSWER"
    )


# =========================================================
# TOP HEADER
# =========================================================

col1, col2 = st.columns(
    [5, 1]
)

with col1:

    st.title("🔷 BIS Sahayak")

    st.caption(
        "AI-powered Intelligent Assistant for Indian "
        "Standards & BIS Services"
    )

with col2:

    st.markdown("### 🇮🇳")

    st.caption(
        "Standards for a Better Tomorrow"
    )


st.divider()


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    # -----------------------------------------------------
    # WELCOME
    # -----------------------------------------------------

    st.header(
        "Welcome to BIS Sahayak"
    )

    st.write(
        "Ask your questions about Indian Standards, "
        "certification, testing, hallmarking and other "
        "BIS services. Get a clear response supported "
        "by relevant official BIS sources."
    )

    st.success(
        "🔎 Find • Understand • Comply"
    )


    # -----------------------------------------------------
    # ASK QUESTION
    # -----------------------------------------------------

    st.subheader(
        "Ask BIS Sahayak"
    )

    with st.form("question_form"):

        question = st.text_input(
            "Your question",
            placeholder=(
                "Example: What is the BIS certification process?"
            )
        )

        submitted = st.form_submit_button(
            "➤ Ask BIS Sahayak",
            type="primary"
        )


    # -----------------------------------------------------
    # EXAMPLE QUESTIONS
    # -----------------------------------------------------

    st.subheader(
        "Try asking something like:"
    )

    col1, col2, col3 = st.columns(3)

    selected_question = None

    with col1:

        if st.button(
            "📄 Find the Indian Standard for my product",
            use_container_width=True
        ):

            selected_question = (
                "How do I find the Indian Standard for my product?"
            )

    with col2:

        if st.button(
            "✓ What is the BIS certification process?",
            use_container_width=True
        ):

            selected_question = (
                "What is the BIS certification process?"
            )

    with col3:

        if st.button(
            "◇ How can I verify HUID?",
            use_container_width=True
        ):

            selected_question = (
                "How can I verify HUID?"
            )


    col4, col5, col6 = st.columns(3)

    with col4:

        if st.button(
            "⚗ Find BIS testing laboratories",
            use_container_width=True
        ):

            selected_question = (
                "How do I find BIS testing laboratories?"
            )

    with col5:

        if st.button(
            "📚 What is Know Your Standard?",
            use_container_width=True
        ):

            selected_question = (
                "What is available in Know Your Standard?"
            )

    with col6:

        if st.button(
            "🔎 Find information about BIS standards",
            use_container_width=True
        ):

            selected_question = (
                "What information is available on BIS standards?"
            )


    # -----------------------------------------------------
    # RUN QUESTION
    # -----------------------------------------------------

    if selected_question:

        question = selected_question
        submitted = True


    if submitted and question.strip():

        question = question.strip()

        results = retrieve(
            question,
            3
        )

        answer = generate_answer(
            question,
            results
        )


        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.subheader(
            "BIS Sahayak Response"
        )

        with st.container(border=True):

            st.caption(
                "GROUNDED ANSWER"
            )

            st.write(
                answer
            )


        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.subheader(
            "Evidence & Official BIS Sources"
        )

        if results:

            for doc, score in results:

                with st.container(border=True):

                    st.markdown(
                        f"### {doc['title']}"
                    )

                    st.caption(
                        f"Relevance score: {score:.2f}"
                    )

                    st.write(
                        doc["content"]
                    )

                    st.link_button(
                        "🔗 Open Official BIS Source",
                        doc["url"]
                    )

        else:

            st.warning(
                "No matching evidence was found."
            )


    # -----------------------------------------------------
    # HOW IT WORKS
    # -----------------------------------------------------

    st.subheader(
        "How BIS Sahayak Works"
    )

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("### 1️⃣ Ask")

        st.write(
            "Enter your question in natural language."
        )

    with step2:

        st.markdown("### 2️⃣ Retrieve")

        st.write(
            "Find relevant BIS information."
        )

    with step3:

        st.markdown("### 3️⃣ Rank")

        st.write(
            "Select the most relevant sources."
        )


    step4, step5, step6 = st.columns(3)

    with step4:

        st.markdown("### 4️⃣ Ground")

        st.write(
            "Verify the response against authorized BIS data."
        )

    with step5:

        st.markdown("### 5️⃣ Cite")

        st.write(
            "Show official BIS sources and references."
        )

    with step6:

        st.markdown("### 6️⃣ Answer")

        st.write(
            "Provide a clear evidence-backed response."
        )


    st.info(
        "🛡️ Evidence notice: This prototype uses a curated "
        "BIS knowledge base. Always verify the latest "
        "information on official BIS portals before making "
        "compliance or certification decisions."
    )


# =========================================================
# STANDARDS
# =========================================================

elif page == "📄 Standards":

    st.header(
        "📄 Indian Standards"
    )

    st.write(
        "Explore BIS resources for finding and "
        "understanding Indian Standards."
    )

    standard_docs = [
        d for d in DOCS
        if d.get("category") == "standards"
    ]

    for doc in standard_docs:

        with st.container(border=True):

            st.subheader(
                doc["title"]
            )

            st.write(
                doc["content"]
            )

            st.link_button(
                "🔗 Open Official BIS Source",
                doc["url"]
            )


# =========================================================
# CERTIFICATION
# =========================================================

elif page == "✓ Certification":

    st.header(
        "✓ BIS Certification"
    )

    st.write(
        "Understand BIS product certification, "
        "conformity assessment and licence-related "
        "information."
    )

    certification_docs = [
        d for d in DOCS
        if d.get("category") == "certification"
    ]

    for doc in certification_docs:

        with st.container(border=True):

            st.subheader(
                doc["title"]
            )

            st.write(
                doc["content"]
            )

            st.link_button(
                "🔗 Open Official BIS Source",
                doc["url"]
            )


# =========================================================
# HALLMARKING
# =========================================================

elif page == "◇ Hallmarking":

    st.header(
        "◇ Hallmarking"
    )

    st.write(
        "Find BIS information related to hallmarking "
        "and HUID verification."
    )

    hallmark_docs = [
        d for d in DOCS
        if d.get("category") == "hallmarking"
    ]

    for doc in hallmark_docs:

        with st.container(border=True):

            st.subheader(
                doc["title"]
            )

            st.write(
                doc["content"]
            )

            st.link_button(
                "🔗 Open Official BIS Source",
                doc["url"]
            )


# =========================================================
# TESTING & LABS
# =========================================================

elif page == "⚗ Testing & Labs":

    st.header(
        "⚗ Testing & Labs"
    )

    st.write(
        "Access BIS laboratory and testing-related "
        "information available in the knowledge base."
    )

    lab_docs = [
        d for d in DOCS
        if d.get("category") == "testing"
    ]

    for doc in lab_docs:

        with st.container(border=True):

            st.subheader(
                doc["title"]
            )

            st.write(
                doc["content"]
            )

            st.link_button(
                "🔗 Open Official BIS Source",
                doc["url"]
            )


# =========================================================
# ABOUT BIS
# =========================================================

elif page == "ⓘ About BIS":

    st.header(
        "ⓘ About BIS"
    )

    st.write(
        "Learn about the Bureau of Indian Standards "
        "and access official BIS information."
    )

    about_docs = [
        d for d in DOCS
        if d.get("category") == "about"
    ]

    for doc in about_docs:

        with st.container(border=True):

            st.subheader(
                doc["title"]
            )

            st.write(
                doc["content"]
            )

            st.link_button(
                "🔗 Open Official BIS Source",
                doc["url"]
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "BIS Sahayak • AI-powered Intelligent Assistant "
    "for Indian Standards & BIS Services"
)

st.caption(
    "Prototype information should be verified against "
    "the latest official BIS source."
)
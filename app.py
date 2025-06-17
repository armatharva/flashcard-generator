# FORCE CACHE CLEAR - Updated 2025-01-17 17:45 UTC
# This timestamp forces Streamlit Cloud to use the new version without caching

import streamlit as st
from transformers import pipeline
from utils import extract_text_from_file, extract_text_from_image
from chatbot import Chatbot
from translations import LANGUAGES, get_translation, translate_text, get_translator
import csv
import io
import random
from fpdf import FPDF
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import re
import json

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Notes to Flashcards AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add custom CSS for the chat button
st.markdown("""
<style>
.chat-button {
    background-color: #4CAF50;
    color: white;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    border: none;
    border-radius: 4px;
    min-width: 80px;
    text-align: center;
    cursor: pointer;
    display: inline-block;
    white-space: nowrap;
    transition: background-color 0.3s;
}
.chat-button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_translator():
    return get_translator()

def translate_text(text, dest_lang):
    """Translate text to the specified language using Google Translate."""
    if dest_lang == "en":
        return text
    try:
        translator = get_translator()
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original text.")
        return text

# FAQ dictionary in English (base language)
FAQ_DICT_EN = {
    "which languages are supported?": "We support the top 20 languages + Telugu.",
    "can i ask follow-up questions?": "Yes! Use the chatbot to ask follow-ups.",
    "how do i upload my notes?": "Click the 'Choose file' button to upload notes.",
    "can i type notes instead of uploading?": "Yes, you can paste or type them.",
    "how do i get flashcards?": "After uploading or pasting notes, click 'Generate Flashcards'.",
    "can i choose how many flashcards i want?": "Yes, use the number selector.",
    "what if my notes are very long?": "The summarizer breaks notes into chunks.",
    "is this service free?": "Yes, the website and all features are free!",
    "how do i contact support?": "For now, please send feedback to our GitHub repo."
}

@st.cache_data(ttl=3600)  # Cache translations for 1 hour
def get_faq_dict_translated(lang_code):
    """Get FAQ dictionary translated to the specified language."""
    if lang_code == "en":
        return FAQ_DICT_EN
    return {q: translate_text(a, lang_code) for q, a in FAQ_DICT_EN.items()}

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_flashcard_generator():
    """Load the flashcard generator model with optimized settings."""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=8,  # Increased batch size
        model_kwargs={
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
        }
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_cached_flashcards(text: str, num_cards: int = 5) -> List[Dict[str, str]]:
    """Cached version of flashcard generation for frequently used content."""
    return generate_flashcards_batch(text, num_cards)

def generate_flashcards_batch(text: str, num_cards: int = 5) -> List[Dict[str, str]]:
    """Generate multiple flashcards in a single batch with improved prompting."""
    generator = load_flashcard_generator()
    
    # Enhanced prompt for better quality flashcards with explanations
    prompt = f"""Generate {num_cards} high-quality flashcards from this text. Follow these guidelines:
    1. Questions should be clear, specific, and test understanding
    2. Answers should be concise but complete
    3. Include a detailed explanation for each answer
    4. Include a mix of:
       - Definition questions
       - Concept explanation questions
       - Application questions
       - Comparison questions
       - Multiple choice questions (with 4 options)
    
    Text: {text}
    
    Format each flashcard as:
    Q: [question]
    A: [answer]
    EXPLANATION: [detailed explanation]
    DIFFICULTY: [easy/medium/hard]
    TYPE: [definition/concept/application/comparison/multiple_choice]
    OPTIONS: [for multiple choice questions only, format as JSON array]
    """
    
    # Generate with optimized parameters
    response = generator(
        prompt,
        max_length=1024,  # Increased for explanations
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )[0]['generated_text']
    
    # Parse the response into flashcards with enhanced metadata
    flashcards = []
    current_card = {
        "question": None,
        "answer": None,
        "explanation": None,
        "difficulty": "medium",
        "type": "concept",
        "options": None
    }
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('Q:'):
            if current_card["question"]:  # Save previous card
                flashcards.append(current_card.copy())
            current_card = {
                "question": line[2:].strip(),
                "answer": None,
                "explanation": None,
                "difficulty": "medium",
                "type": "concept",
                "options": None
            }
        elif line.startswith('A:'):
            current_card["answer"] = line[2:].strip()
        elif line.startswith('EXPLANATION:'):
            current_card["explanation"] = line[12:].strip()
        elif line.startswith('DIFFICULTY:'):
            current_card["difficulty"] = line[11:].strip().lower()
        elif line.startswith('TYPE:'):
            current_card["type"] = line[6:].strip().lower()
        elif line.startswith('OPTIONS:'):
            try:
                current_card["options"] = json.loads(line[8:].strip())
            except:
                current_card["options"] = None
    
    # Add the last card if exists
    if current_card["question"] and current_card["answer"]:
        flashcards.append(current_card)
    
    return flashcards

def generate_flashcards(text: str) -> List[Dict[str, str]]:
    """Generate flashcards from text using optimized batch processing and caching."""
    lang = st.session_state.selected_lang_code
    
    # Check cache first for exact text match
    cache_key = f"{text[:1000]}_{lang}"  # Use first 1000 chars as key
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Split text into optimized chunks
    chunks = chunk_text(text, max_chunk_size=1500, overlap=200)  # Increased chunk size and overlap
    all_flashcards = []
    
    # Process chunks in parallel with progress tracking
    total_chunks = len(chunks)
    processed_chunks = 0
    
    with ThreadPoolExecutor(max_workers=min(4, total_chunks)) as executor:
        # Generate flashcards for each chunk
        future_to_chunk = {
            executor.submit(generate_flashcards_batch, chunk): chunk 
            for chunk in chunks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            try:
                chunk_flashcards = future.result()
                all_flashcards.extend(chunk_flashcards)
                processed_chunks += 1
                
                # Update progress
                if 'progress_bar' in st.session_state:
                    progress = 30 + (60 * processed_chunks / total_chunks)
                    st.session_state.progress_bar.progress(min(90, int(progress)))
                
            except Exception as e:
                st.error(f"Error processing chunk: {str(e)}")
    
    # Remove duplicates while preserving order and difficulty
    seen = set()
    unique_flashcards = []
    for card in all_flashcards:
        card_key = (card['question'], card['answer'])
        if card_key not in seen:
            seen.add(card_key)
            unique_flashcards.append(card)
    
    # Sort by difficulty
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    unique_flashcards.sort(key=lambda x: difficulty_order.get(x.get('difficulty', 'medium'), 1))
    
    # Cache the results
    st.session_state[cache_key] = unique_flashcards
    
    # Translate flashcards if needed
    if lang != "en":
        translated_flashcards = []
        for card in unique_flashcards:
            translated_flashcards.append({
                "question": translate_text(card['question'], lang),
                "answer": translate_text(card['answer'], lang),
                "difficulty": card.get('difficulty', 'medium')
            })
        return translated_flashcards
    
    return unique_flashcards

def render_flip_cards(flashcards: List[Dict[str, str]], lang_code: str):
    """Render flashcards with difficulty indicators and improved styling."""
    html = """
    <style>
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        padding: 20px;
    }

    .flip-card {
        background-color: transparent;
        width: 300px;
        height: 200px;
        perspective: 1000px;
        margin-bottom: 20px;
    }

    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 15px;
    }

    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }

    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        box-sizing: border-box;
    }

    .flip-card-front {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #2c3e50;
        border: 1px solid #e9ecef;
    }

    .flip-card-back {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        transform: rotateY(180deg);
        border: 1px solid #dee2e6;
    }

    .card-number {
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 14px;
        color: #6c757d;
    }

    .card-content {
        font-size: 16px;
        line-height: 1.5;
        overflow-y: auto;
        max-height: 140px;
        width: 100%;
        padding: 10px;
    }

    .card-label {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom scrollbar for card content */
    .card-content::-webkit-scrollbar {
        width: 6px;
    }

    .card-content::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }

    .card-content::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }

    .card-content::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .flip-card {
            width: 100%;
            max-width: 300px;
        }
    }
    </style>
    <div class="card-container">
    """

    # Add difficulty indicators to the cards
    difficulty_colors = {
        "easy": "#4CAF50",  # Green
        "medium": "#FFC107",  # Yellow
        "hard": "#F44336"  # Red
    }
    
    for i, card in enumerate(flashcards):
        difficulty = card.get('difficulty', 'medium')
        difficulty_color = difficulty_colors.get(difficulty, "#FFC107")
        
        # Add difficulty indicator to card HTML
        html += f"""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front" style="border-left: 5px solid {difficulty_color}">
                    <div class="card-number">
                        {translate_text('Card', lang_code)} {i+1}
                        <span class="difficulty-badge" style="background-color: {difficulty_color}">
                            {translate_text(difficulty.capitalize(), lang_code)}
                        </span>
                    </div>
                    <div class="card-label">{translate_text('Question', lang_code)}</div>
                    <div class="card-content">{card['question']}</div>
                </div>
                <div class="flip-card-back" style="border-left: 5px solid {difficulty_color}">
                    <div class="card-number">
                        {translate_text('Card', lang_code)} {i+1}
                        <span class="difficulty-badge" style="background-color: {difficulty_color}">
                            {translate_text(difficulty.capitalize(), lang_code)}
                        </span>
                    </div>
                    <div class="card-label">{translate_text('Answer', lang_code)}</div>
                    <div class="card-content">{card['answer']}</div>
                </div>
            </div>
        </div>
        """

    html += "</div>"
    
    # Add instructions
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px; color: #666;'>
        {translate_text('💡 Hover over a card to flip it and see the answer', lang_code)}
    </div>
    """, unsafe_allow_html=True)
    
    # Add difficulty legend
    html += """
    <div class="difficulty-legend">
        <span style="color: #4CAF50">●</span> Easy
        <span style="color: #FFC107">●</span> Medium
        <span style="color: #F44336">●</span> Hard
    </div>
    """
    
    # Render the cards
    st.markdown(html, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot(FAQ_DICT_EN)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_notes" not in st.session_state:
        st.session_state.current_notes = None
    if "selected_lang_code" not in st.session_state:
        st.session_state.selected_lang_code = "en"
    if "flashcard_history" not in st.session_state:
        st.session_state.flashcard_history = []
    if "study_stats" not in st.session_state:
        st.session_state.study_stats = {
            "matching_games_played": 0,
            "matching_games_won": 0,
            "practice_tests_taken": 0,
            "practice_tests_passed": 0,
            "total_flashcards_created": 0
        }

def save_flashcards_to_history(flashcards, source="generated"):
    """Save flashcards to history with metadata."""
    timestamp = datetime.now().isoformat()
    history_entry = {
        "timestamp": timestamp,
        "flashcards": flashcards,
        "source": source,
        "language": st.session_state.selected_lang_code
    }
    st.session_state.flashcard_history.append(history_entry)
    st.session_state.study_stats["total_flashcards_created"] += len(flashcards)

def render_flashcard_history():
    """Render the flashcard history section."""
    st.markdown("### " + translate_text("📚 Flashcard History", st.session_state.selected_lang_code))
    
    if not st.session_state.flashcard_history:
        st.info(translate_text("No flashcard history yet. Generate some flashcards to see them here!", st.session_state.selected_lang_code))
        return
    
    # Group history by date
    history_by_date = {}
    for entry in reversed(st.session_state.flashcard_history):
        date = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d")
        if date not in history_by_date:
            history_by_date[date] = []
        history_by_date[date].append(entry)
    
    # Display history
    for date, entries in history_by_date.items():
        with st.expander(f"📅 {date} ({len(entries)} sets)"):
            for entry in entries:
                time = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{time}** - {translate_text(entry['source'], st.session_state.selected_lang_code)}")
                    st.markdown(f"*{translate_text('Language:', st.session_state.selected_lang_code)} {entry['language']}*")
                
                with col2:
                    if st.button(
                        translate_text("View Cards", st.session_state.selected_lang_code),
                        key=f"view_{entry['timestamp']}"
                    ):
                        st.session_state.current_flashcards = entry["flashcards"]
                        st.experimental_rerun()
                
                with col3:
                    if st.button(
                        translate_text("Delete", st.session_state.selected_lang_code),
                        key=f"delete_{entry['timestamp']}"
                    ):
                        st.session_state.flashcard_history.remove(entry)
                        st.experimental_rerun()

def render_study_stats():
    """Render study statistics."""
    st.markdown("### " + translate_text("📊 Study Statistics", st.session_state.selected_lang_code))
    
    stats = st.session_state.study_stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            translate_text("Total Flashcards", st.session_state.selected_lang_code),
            stats["total_flashcards_created"]
        )
    
    with col2:
        matching_win_rate = (
            (stats["matching_games_won"] / stats["matching_games_played"] * 100)
            if stats["matching_games_played"] > 0
            else 0
        )
        st.metric(
            translate_text("Matching Game Win Rate", st.session_state.selected_lang_code),
            f"{matching_win_rate:.1f}%"
        )
    
    with col3:
        practice_pass_rate = (
            (stats["practice_tests_passed"] / stats["practice_tests_taken"] * 100)
            if stats["practice_tests_taken"] > 0
            else 0
        )
        st.metric(
            translate_text("Practice Test Pass Rate", st.session_state.selected_lang_code),
            f"{practice_pass_rate:.1f}%"
        )

def update_study_stats(game_type, won=False):
    """Update study statistics."""
    if game_type == "matching":
        st.session_state.study_stats["matching_games_played"] += 1
        if won:
            st.session_state.study_stats["matching_games_won"] += 1
    elif game_type == "practice":
        st.session_state.study_stats["practice_tests_taken"] += 1
        if won:
            st.session_state.study_stats["practice_tests_passed"] += 1

def render_chat_message(message, is_user=True):
    """Render a chat message with enhanced styling."""
    if is_user:
        st.markdown(f"""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 15px; 
                        margin: 10px 0; max-width: 80%; margin-left: auto;'>
                <div style='color: #1976d2; font-weight: bold; margin-bottom: 5px;'>
                    {translate_text('You', st.session_state.selected_lang_code)}
                </div>
                <div style='color: #333;'>{message}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: #f5f5f5; padding: 15px; border-radius: 15px; 
                        margin: 10px 0; max-width: 80%;'>
                <div style='color: #666; font-weight: bold; margin-bottom: 5px;'>
                    {translate_text('AI Assistant', st.session_state.selected_lang_code)}
                </div>
                <div style='color: #333;'>{message}</div>
            </div>
        """, unsafe_allow_html=True)

def process_chat_input(user_input):
    """Process user input and generate appropriate response."""
    # Get translated FAQ dictionary for current language
    faq_dict = get_faq_dict_translated(st.session_state.selected_lang_code)
    
    # Check if it's a question about the app using the callable chatbot
    response = st.session_state.chatbot(user_input, faq_dict)
    
    # If it's not a FAQ question, try to generate flashcards
    if response == translate_text("I'm sorry, I don't know the answer to that. Try asking something else!", st.session_state.selected_lang_code):
        if st.session_state.current_notes:
            try:
                with st.spinner(translate_text("Generating flashcards...", st.session_state.selected_lang_code)):
                    flashcards = generate_flashcards(st.session_state.current_notes)
                    response = translate_text("Here are some flashcards based on your notes:\n\n", st.session_state.selected_lang_code) + "\n\n".join([f"{card['question']}\n{card['answer']}" for card in flashcards])
            except Exception as e:
                st.error(translate_text(f"Error generating flashcards: {str(e)}", st.session_state.selected_lang_code))
                response = translate_text("I had trouble generating flashcards. Please try again.", st.session_state.selected_lang_code)
        else:
            response = translate_text("I can help you generate flashcards! Please upload or enter some notes first.", st.session_state.selected_lang_code)
    
    return response

def matching_game(flashcards, lang_code):
    """Interactive matching game for flashcards."""
    st.header(translate_text("🎯 Flashcard Matching Game", lang_code))

    # Extract questions and answers from flashcard dictionaries
    questions = [card['question'] for card in flashcards]
    answers = [card['answer'] for card in flashcards]
    
    # Filter out empty answers
    valid_pairs = [(q, a) for q, a in zip(questions, answers) if a.strip()]
    if not valid_pairs:
        st.warning(translate_text("No valid question-answer pairs found for the matching game.", lang_code))
        return

    questions, answers = zip(*valid_pairs)
    
    # Shuffle answers for matching
    shuffled_answers = list(answers)
    random.shuffle(shuffled_answers)

    st.write(translate_text("Match each question with the correct answer:", lang_code))

    # Initialize or get game state
    if "matching_game_state" not in st.session_state:
        st.session_state.matching_game_state = {
            "user_matches": {},
            "score": None,
            "shuffled_answers": shuffled_answers
        }

    # Display matching interface
    user_matches = {}
    for i, question in enumerate(questions):
        user_matches[i] = st.selectbox(
            f"Q{i+1}: {question}",
            options=[""] + st.session_state.matching_game_state["shuffled_answers"],
            key=f"match_{i}",
            help=translate_text("Select the matching answer", lang_code)
        )

    col1, col2 = st.columns([1, 3])
    with col1:
        check_button = st.button(
            translate_text("Check Answers", lang_code),
            type="primary"
        )
    with col2:
        reset_button = st.button(
            translate_text("Reset Game", lang_code),
            type="secondary"
        )

    if reset_button:
        st.session_state.matching_game_state = {
            "user_matches": {},
            "score": None,
            "shuffled_answers": random.sample(shuffled_answers, len(shuffled_answers))
        }
        st.experimental_rerun()

    if check_button:
        score = 0
        feedback = []
        for i, question in enumerate(questions):
            if user_matches[i] == answers[i]:
                score += 1
                feedback.append(f"✅ Q{i+1}: {translate_text('Correct!', lang_code)}")
            else:
                correct_answer = answers[i]
                feedback.append(f"❌ Q{i+1}: {translate_text('Incorrect. Correct answer:', lang_code)} {correct_answer}")

        st.session_state.matching_game_state["score"] = score
        
        # Update study stats
        update_study_stats("matching", score == len(questions))
        
        # Display score and feedback
        st.markdown("---")
        st.markdown(f"### {translate_text('Results:', lang_code)}")
        st.success(translate_text(f"You got {score} out of {len(questions)} correct!", lang_code))
        
        # Show detailed feedback
        with st.expander(translate_text("View Detailed Feedback", lang_code)):
            for item in feedback:
                st.write(item)

        if score == len(questions):
            st.balloons()
            st.success(translate_text("🎉 Perfect score! Well done!", lang_code))

def practice_test(flashcards, lang_code):
    """Interactive practice test for flashcards."""
    st.header(translate_text("📝 Practice Test", lang_code))

    # Initialize test state if not exists
    if "practice_test_state" not in st.session_state:
        st.session_state.practice_test_state = {
            "answers": {},
            "checked": {},
            "score": None,
            "show_score": False
        }

    # Prepare questions and answers
    questions = []
    answers = []
    for card in flashcards:
        if ":" in card:
            q, a = card.split(":", 1)
            questions.append(q.strip())
            answers.append(a.strip())
        else:
            questions.append(card.strip())
            answers.append("")

    # Filter out empty answers
    valid_pairs = [(q, a) for q, a in zip(questions, answers) if a]
    if not valid_pairs:
        st.warning(translate_text("No valid question-answer pairs found for the practice test.", lang_code))
        return

    questions, answers = zip(*valid_pairs)
    total_questions = len(questions)

    # Test interface
    st.write(translate_text("Answer each question and check your answers:", lang_code))
    
    # Track correct answers
    correct_answers = 0
    
    # Display questions and answer inputs
    for i, (question, answer) in enumerate(zip(questions, answers)):
        st.markdown("---")
        st.markdown(f"### {translate_text('Question', lang_code)} {i+1}")
        st.markdown(f"**{question}**")
        
        # Answer input
        user_answer = st.text_input(
            translate_text("Your answer:", lang_code),
            key=f"test_{i}",
            value=st.session_state.practice_test_state["answers"].get(i, ""),
            help=translate_text("Type your answer and click 'Check Answer'", lang_code)
        )
        
        # Store answer
        st.session_state.practice_test_state["answers"][i] = user_answer
        
        # Check answer button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                translate_text("Check Answer", lang_code),
                key=f"check_{i}",
                type="primary" if not st.session_state.practice_test_state["checked"].get(i) else "secondary"
            ):
                st.session_state.practice_test_state["checked"][i] = True
                if user_answer.strip().lower() == answer.strip().lower():
                    st.success(translate_text("✅ Correct!", lang_code))
                    correct_answers += 1
                else:
                    st.error(f"{translate_text('❌ Incorrect!', lang_code)} {translate_text('Correct answer:', lang_code)} {answer.strip()}")
        
        # Show feedback if answer was checked
        if st.session_state.practice_test_state["checked"].get(i):
            if user_answer.strip().lower() == answer.strip().lower():
                st.success(translate_text("✅ Correct!", lang_code))
            else:
                st.error(f"{translate_text('❌ Incorrect!', lang_code)} {translate_text('Correct answer:', lang_code)} {answer.strip()}")

    # Score and reset section
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(
            translate_text("Show Final Score", lang_code),
            type="primary",
            use_container_width=True
        ):
            st.session_state.practice_test_state["show_score"] = True
            st.session_state.practice_test_state["score"] = correct_answers
    
    with col2:
        if st.button(
            translate_text("Reset Test", lang_code),
            type="secondary",
            use_container_width=True
        ):
            st.session_state.practice_test_state = {
                "answers": {},
                "checked": {},
                "score": None,
                "show_score": False
            }
            st.experimental_rerun()

    # Display final score if requested
    if st.session_state.practice_test_state["show_score"]:
        st.markdown("### " + translate_text("Test Results", lang_code))
        score = st.session_state.practice_test_state["score"]
        percentage = (score / total_questions) * 100
        
        # Update study stats
        update_study_stats("practice", percentage >= 70)
        
        # Score display with color coding
        if percentage >= 90:
            st.success(f"🎉 {translate_text('Excellent!', lang_code)} {score}/{total_questions} ({percentage:.1f}%)")
        elif percentage >= 70:
            st.info(f"👍 {translate_text('Good job!', lang_code)} {score}/{total_questions} ({percentage:.1f}%)")
        else:
            st.warning(f"📚 {translate_text('Keep practicing!', lang_code)} {score}/{total_questions} ({percentage:.1f}%)")
        
        if score == total_questions:
            st.balloons()
            st.success(translate_text("🎉 Perfect score! Well done!", lang_code))

def quiz_mode(flashcards: List[Dict[str, str]], lang_code: str):
    """Interactive quiz mode for flashcards with enhanced multiple choice."""
    st.header(translate_text("📝 Quiz Mode", lang_code))
    
    # Initialize quiz state with enhanced features
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {
            "current_index": 0,
            "score": 0,
            "answers": {},
            "show_explanation": False,
            "quiz_completed": False,
            "shuffled_options": {},  # Store shuffled options for multiple choice
            "time_started": None,
            "time_elapsed": 0
        }
    
    # Start timer if not started
    if not st.session_state.quiz_state["time_started"]:
        st.session_state.quiz_state["time_started"] = datetime.now()
    
    # Get current flashcard
    current_card = flashcards[st.session_state.quiz_state["current_index"]]
    
    # Generate shuffled options for multiple choice if needed
    if current_card["type"] == "multiple_choice" and current_card["options"] is None:
        if st.session_state.quiz_state["current_index"] not in st.session_state.quiz_state["shuffled_options"]:
            # Get wrong answers from other cards
            wrong_answers = [
                card["answer"] for card in flashcards 
                if card["answer"] != current_card["answer"]
            ]
            # Shuffle and take 3 wrong answers
            wrong_answers = random.sample(wrong_answers, min(3, len(wrong_answers)))
            # Combine with correct answer and shuffle
            options = wrong_answers + [current_card["answer"]]
            random.shuffle(options)
            st.session_state.quiz_state["shuffled_options"][st.session_state.quiz_state["current_index"]] = options
    
    # Display quiz info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            translate_text("Score", lang_code),
            f"{st.session_state.quiz_state['score']}/{len(flashcards)}"
        )
    with col2:
        # Calculate time elapsed
        if not st.session_state.quiz_state["quiz_completed"]:
            time_elapsed = (datetime.now() - st.session_state.quiz_state["time_started"]).total_seconds()
            st.session_state.quiz_state["time_elapsed"] = time_elapsed
        st.metric(
            translate_text("Time", lang_code),
            f"{int(st.session_state.quiz_state['time_elapsed'])}s"
        )
    with col3:
        st.metric(
            translate_text("Progress", lang_code),
            f"{st.session_state.quiz_state['current_index'] + 1}/{len(flashcards)}"
        )
    
    # Display progress bar
    progress = (st.session_state.quiz_state["current_index"] + 1) / len(flashcards)
    st.progress(progress)
    
    # Display question with enhanced formatting
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;'>
        <h3 style='color: #2c3e50;'>{current_card['question']}</h3>
        <p style='color: #666; font-size: 0.9em;'>
            {translate_text('Type:', lang_code)} {translate_text(current_card['type'].replace('_', ' ').title(), lang_code)} | 
            {translate_text('Difficulty:', lang_code)} {translate_text(current_card['difficulty'].title(), lang_code)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display answer options based on question type
    if current_card["type"] == "multiple_choice":
        options = st.session_state.quiz_state["shuffled_options"].get(
            st.session_state.quiz_state["current_index"],
            current_card.get("options", [])
        )
        user_answer = st.radio(
            translate_text("Select your answer:", lang_code),
            options=options,
            key=f"quiz_{st.session_state.quiz_state['current_index']}",
            label_visibility="collapsed"
        )
    else:
        user_answer = st.text_input(
            translate_text("Your answer:", lang_code),
            key=f"quiz_{st.session_state.quiz_state['current_index']}",
            label_visibility="collapsed"
        )
    
    # Store answer
    st.session_state.quiz_state["answers"][st.session_state.quiz_state["current_index"]] = user_answer
    
    # Navigation and answer checking
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button(
            translate_text("⏮️ Previous", lang_code),
            disabled=st.session_state.quiz_state["current_index"] == 0,
            use_container_width=True
        ):
            st.session_state.quiz_state["current_index"] -= 1
            st.session_state.quiz_state["show_explanation"] = False
            st.experimental_rerun()
    
    with col2:
        if st.button(
            translate_text("Check Answer", lang_code),
            type="primary",
            use_container_width=True
        ):
            st.session_state.quiz_state["show_explanation"] = True
            # Check answer with enhanced matching
            if current_card["type"] == "multiple_choice":
                is_correct = user_answer == current_card["answer"]
            else:
                # Use fuzzy matching for text answers
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, 
                    user_answer.lower().strip(),
                    current_card["answer"].lower().strip()
                ).ratio()
                is_correct = similarity > 0.8  # 80% similarity threshold
            
            if is_correct:
                st.session_state.quiz_state["score"] += 1
                st.success(translate_text("✅ Correct!", lang_code))
            else:
                st.error(translate_text("❌ Incorrect!", lang_code))
    
    with col3:
        if st.button(
            translate_text("Next ⏭️", lang_code),
            disabled=st.session_state.quiz_state["current_index"] == len(flashcards) - 1,
            use_container_width=True
        ):
            st.session_state.quiz_state["current_index"] += 1
            st.session_state.quiz_state["show_explanation"] = False
            st.experimental_rerun()
    
    # Show explanation with enhanced formatting
    if st.session_state.quiz_state["show_explanation"]:
        with st.expander(translate_text("📚 Explanation", lang_code)):
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c3e50;'>{translate_text('Explanation:', lang_code)}</h4>
                <p style='color: #333;'>{current_card['explanation']}</p>
                <hr style='margin: 10px 0;'>
                <h4 style='color: #2c3e50;'>{translate_text('Correct Answer:', lang_code)}</h4>
                <p style='color: #333;'>{current_card['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quiz completion
    if st.session_state.quiz_state["current_index"] == len(flashcards) - 1:
        if st.button(translate_text("Finish Quiz", lang_code), type="primary", use_container_width=True):
            st.session_state.quiz_state["quiz_completed"] = True
            st.experimental_rerun()
    
    # Display quiz results with enhanced UI
    if st.session_state.quiz_state["quiz_completed"]:
        st.markdown("---")
        st.markdown("### " + translate_text("🎯 Quiz Results", lang_code))
        
        score = st.session_state.quiz_state["score"]
        total = len(flashcards)
        percentage = (score / total) * 100
        time_taken = st.session_state.quiz_state["time_elapsed"]
        
        # Calculate stats
        avg_time = time_taken / total
        speed = "Fast" if avg_time < 10 else "Medium" if avg_time < 20 else "Slow"
        
        # Display results in a nice grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                translate_text("Score", lang_code),
                f"{score}/{total}",
                f"{percentage:.1f}%"
            )
        
        with col2:
            st.metric(
                translate_text("Time Taken", lang_code),
                f"{int(time_taken)}s",
                f"{int(avg_time)}s {translate_text('per question', lang_code)}"
            )
        
        with col3:
            st.metric(
                translate_text("Speed", lang_code),
                translate_text(speed, lang_code)
            )
        
        # Display score with color coding and emoji
        if percentage >= 90:
            st.success(f"🎉 {translate_text('Excellent!', lang_code)} {score}/{total} ({percentage:.1f}%)")
        elif percentage >= 70:
            st.info(f"👍 {translate_text('Good job!', lang_code)} {score}/{total} ({percentage:.1f}%)")
        else:
            st.warning(f"📚 {translate_text('Keep practicing!', lang_code)} {score}/{total} ({percentage:.1f}%)")
        
        # Show detailed results with enhanced formatting
        with st.expander(translate_text("View Detailed Results", lang_code)):
            for i, card in enumerate(flashcards):
                user_ans = st.session_state.quiz_state["answers"].get(i, "")
                is_correct = (user_ans.lower().strip() == card["answer"].lower().strip())
                
                st.markdown(f"""
                <div style='background-color: {'#e8f5e9' if is_correct else '#ffebee'}; 
                            padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: #2c3e50;'>{translate_text('Question', lang_code)} {i+1}:</h4>
                    <p style='color: #333;'>{card['question']}</p>
                    <hr style='margin: 10px 0;'>
                    <p><strong>{translate_text('Your Answer:', lang_code)}</strong> {user_ans}</p>
                    <p><strong>{translate_text('Correct Answer:', lang_code)}</strong> {card['answer']}</p>
                    <p><strong>{translate_text('Explanation:', lang_code)}</strong> {card['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(translate_text("🔄 Retake Quiz", lang_code), use_container_width=True):
                st.session_state.quiz_state = {
                    "current_index": 0,
                    "score": 0,
                    "answers": {},
                    "show_explanation": False,
                    "quiz_completed": False,
                    "shuffled_options": {},
                    "time_started": None,
                    "time_elapsed": 0
                }
                st.experimental_rerun()
        
        with col2:
            if st.button(translate_text("📊 Save Results", lang_code), use_container_width=True):
                # Save quiz results to history
                quiz_result = {
                    "timestamp": datetime.now().isoformat(),
                    "score": score,
                    "total": total,
                    "percentage": percentage,
                    "time_taken": time_taken,
                    "avg_time": avg_time,
                    "speed": speed,
                    "answers": st.session_state.quiz_state["answers"]
                }
                
                if "quiz_history" not in st.session_state:
                    st.session_state.quiz_history = []
                
                st.session_state.quiz_history.append(quiz_result)
                st.success(translate_text("Results saved to history!", lang_code))

def render_language_selector():
    """Render a user-friendly language selector in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {translate_text('🌐 Language Settings', st.session_state.selected_lang_code)}")
    
    # Language groups for better organization
    language_groups = {
        translate_text("Popular Languages", st.session_state.selected_lang_code): [
            "English", "Español", "中文", "हिंदी", "Français", "Deutsch", "日本語"
        ],
        translate_text("More Languages", st.session_state.selected_lang_code): [
            "한국어", "Русский", "Português", "العربية", "Italiano", "বাংলা", "اردو",
            "Türkçe", "Tiếng Việt", "Polski", "Nederlands", "Bahasa Indonesia",
            "Svenska", "తెలుగు"
        ]
    }
    
    # Create tabs for language groups
    lang_tabs = st.sidebar.tabs(list(language_groups.keys()))
    
    # Track if language was changed
    language_changed = False
    
    # Populate each tab with language buttons
    for tab, languages in zip(lang_tabs, language_groups.values()):
        with tab:
            cols = st.columns(2)  # 2 columns for better layout
            for i, lang in enumerate(languages):
                with cols[i % 2]:
                    # Create a button for each language
                    if st.button(
                        lang,
                        key=f"lang_{lang}",
                        use_container_width=True,
                        type="primary" if LANGUAGES[lang] == st.session_state.selected_lang_code else "secondary"
                    ):
                        st.session_state.selected_lang_code = LANGUAGES[lang]
                        language_changed = True
    
    # Show current language
    current_lang = next((k for k, v in LANGUAGES.items() if v == st.session_state.selected_lang_code), "English")
    st.sidebar.markdown(f"**{translate_text('Current Language:', st.session_state.selected_lang_code)}** {current_lang}")
    
    # Language change notification
    if language_changed:
        st.sidebar.success(translate_text("Language changed! The interface will update.", st.session_state.selected_lang_code))
        # Clear any cached translations
        if "translator" in st.session_state:
            del st.session_state.translator
        st.experimental_rerun()

def create_flashcards_pdf(flashcards, lang_code):
    """Create a PDF document with flashcards."""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font for title
    pdf.set_font("Arial", "B", 16)
    title = translate_text("Your Flashcards", lang_code)
    pdf.cell(0, 20, title, ln=True, align="C")
    pdf.ln(10)
    
    # Set font for cards
    pdf.set_font("Arial", "", 12)
    
    # Add each flashcard
    for i, card in enumerate(flashcards):
        # Card number
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{translate_text('Card', lang_code)} {i+1}", ln=True)
        pdf.ln(5)
        
        # Question
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{translate_text('Question', lang_code)}:", ln=True)
        pdf.set_font("Arial", "", 12)
        # Handle multi-line questions
        pdf.multi_cell(0, 10, card['question'])
        pdf.ln(5)
        
        # Answer
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{translate_text('Answer', lang_code)}:", ln=True)
        pdf.set_font("Arial", "", 12)
        # Handle multi-line answers
        pdf.multi_cell(0, 10, card['answer'])
        
        # Add separator between cards
        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
    
    # Get PDF as bytes and create BytesIO object
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output = io.BytesIO(pdf_bytes)
    pdf_output.seek(0)
    
    return pdf_output

def render_print_button():
    """Render a button that triggers the browser's print dialog."""
    print_js = """
    <script>
    function printFlashcards() {
        window.print();
    }
    </script>
    <button onclick="printFlashcards()" style="
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    ">🖨️ Print Flashcards</button>
    """
    components.html(print_js, height=50)

def render_feedback_form():
    """Render the feedback form in the sidebar."""
    # Function temporarily disabled to fix expander issues
    pass

def main():
    initialize_session_state()
    
    # Render language selector in sidebar
    render_language_selector()
    
    # Main content
    st.title(translate_text("🧠 Notes to Flashcards AI", st.session_state.selected_lang_code))
    
    # Language-specific welcome message
    st.info(translate_text(
        "Welcome! This app supports multiple languages. Choose your preferred language from the sidebar. "
        "All content, including flashcards and chatbot responses, will be displayed in your selected language.",
        st.session_state.selected_lang_code
    ))

    # Sidebar with enhanced chatbot
    with st.sidebar:
        st.header(translate_text("💬 Interactive Chat", st.session_state.selected_lang_code))
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                render_chat_message(
                    translate_text(message["text"], st.session_state.selected_lang_code),
                    is_user=(message["role"] == "user")
                )
        
        # Chat input
        st.markdown("---")
        user_input = st.text_input(
            translate_text("Type your question here...", st.session_state.selected_lang_code),
            key="chat_input"
        )
        
        # Create a container for the button with custom styling
        button_container = st.container()
        with button_container:
            col1, col2 = st.columns([1, 4])
            with col1:
                # Create a custom styled button using HTML
                button_html = f"""
                <div style="display: flex; justify-content: center; align-items: center; min-width: 80px;">
                    <button class="chat-button" onclick="document.getElementById('send_button_clicked').click()">
                        {translate_text("Send", st.session_state.selected_lang_code)}
                    </button>
                </div>
                """
                st.markdown(button_html, unsafe_allow_html=True)
                # Hidden Streamlit button to handle the click
                send_button_clicked = st.button(
                    translate_text("Send", st.session_state.selected_lang_code),
                    key="send_button_clicked",
                    type="primary",
                    use_container_width=True
                )
        
        # Process input on button click or Enter key
        if (send_button_clicked or user_input) and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "text": user_input
            })
            
            # Generate and add AI response
            with st.spinner(translate_text("Thinking...", st.session_state.selected_lang_code)):
                ai_response = process_chat_input(user_input)
                st.session_state.chat_history.append({
                    "role": "ai",
                    "text": ai_response
                })
            
            # Clear input
            st.session_state.chat_input = ""
            st.experimental_rerun()

    # Add history and stats sections
    st.markdown("---")
    render_study_stats()
    st.markdown("---")
    render_flashcard_history()

    # Footer
    st.markdown("---")
    st.markdown(translate_text("Made with ❤️ using Streamlit, BART, and FLAN-T5", st.session_state.selected_lang_code))

if __name__ == "__main__":
    main() 
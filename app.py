import streamlit as st
import pytesseract
from PIL import Image
import io
import base64
import os
from gtts import gTTS
from typing import Dict, List, Optional
import random
from chatbot import Chatbot
from translations import translate_text, LANGUAGES
import tempfile
import time
from pydub import AudioSegment

# Map our language codes to gTTS language codes and voice options
LANG_MAP = {
    # Major Languages with voice options
    "en": {"code": "en", "voices": ["en-US", "en-GB", "en-AU", "en-IN"]},  # English variants
    "es": {"code": "es", "voices": ["es-ES", "es-MX", "es-AR", "es-CO"]},  # Spanish variants
    "zh": {"code": "zh-cn", "voices": ["zh-CN", "zh-TW", "zh-HK"]},  # Chinese variants
    "hi": {"code": "hi", "voices": ["hi-IN"]},  # Hindi
    "fr": {"code": "fr", "voices": ["fr-FR", "fr-CA", "fr-BE", "fr-CH"]},  # French variants
    "de": {"code": "de", "voices": ["de-DE", "de-AT", "de-CH"]},  # German variants
    "ja": {"code": "ja", "voices": ["ja-JP"]},  # Japanese
    "ko": {"code": "ko", "voices": ["ko-KR"]},  # Korean
    "ru": {"code": "ru", "voices": ["ru-RU"]},  # Russian
    "pt": {"code": "pt", "voices": ["pt-BR", "pt-PT"]},  # Portuguese variants
    "ar": {"code": "ar", "voices": ["ar-SA", "ar-EG", "ar-MA"]},  # Arabic variants
    "it": {"code": "it", "voices": ["it-IT", "it-CH"]},  # Italian variants
    
    # South Asian with voice options
    "bn": {"code": "bn", "voices": ["bn-IN", "bn-BD"]},  # Bengali
    "te": {"code": "te", "voices": ["te-IN"]},  # Telugu
    "ta": {"code": "ta", "voices": ["ta-IN", "ta-SG", "ta-MY"]},  # Tamil
    "mr": {"code": "mr", "voices": ["mr-IN"]},  # Marathi
    "gu": {"code": "gu", "voices": ["gu-IN"]},  # Gujarati
    "kn": {"code": "kn", "voices": ["kn-IN"]},  # Kannada
    "ml": {"code": "ml", "voices": ["ml-IN"]},  # Malayalam
    "pa": {"code": "pa", "voices": ["pa-IN", "pa-PK"]},  # Punjabi
    "ur": {"code": "ur", "voices": ["ur-IN", "ur-PK"]},  # Urdu
    "ne": {"code": "ne", "voices": ["ne-NP"]},  # Nepali
    "si": {"code": "si", "voices": ["si-LK"]},  # Sinhala
    
    # European with voice options
    "nl": {"code": "nl", "voices": ["nl-NL", "nl-BE"]},  # Dutch variants
    "pl": {"code": "pl", "voices": ["pl-PL"]},  # Polish
    "sv": {"code": "sv", "voices": ["sv-SE", "sv-FI"]},  # Swedish variants
    "da": {"code": "da", "voices": ["da-DK"]},  # Danish
    "fi": {"code": "fi", "voices": ["fi-FI"]},  # Finnish
    "no": {"code": "no", "voices": ["nb-NO", "nn-NO"]},  # Norwegian variants
    "el": {"code": "el", "voices": ["el-GR", "el-CY"]},  # Greek variants
    "hu": {"code": "hu", "voices": ["hu-HU"]},  # Hungarian
    "ro": {"code": "ro", "voices": ["ro-RO"]},  # Romanian
    "sk": {"code": "sk", "voices": ["sk-SK"]},  # Slovak
    "uk": {"code": "uk", "voices": ["uk-UA"]},  # Ukrainian
    "bg": {"code": "bg", "voices": ["bg-BG"]},  # Bulgarian
    "hr": {"code": "hr", "voices": ["hr-HR"]},  # Croatian
    "sr": {"code": "sr", "voices": ["sr-RS", "sr-ME"]},  # Serbian variants
    "ca": {"code": "ca", "voices": ["ca-ES", "ca-FR", "ca-AD"]},  # Catalan variants
    "eu": {"code": "eu", "voices": ["eu-ES"]},  # Basque
    "gl": {"code": "gl", "voices": ["gl-ES"]},  # Galician
    "is": {"code": "is", "voices": ["is-IS"]},  # Icelandic
    
    # Middle Eastern with voice options
    "fa": {"code": "fa", "voices": ["fa-IR", "fa-AF"]},  # Persian variants
    "tr": {"code": "tr", "voices": ["tr-TR", "tr-CY"]},  # Turkish variants
    "he": {"code": "he", "voices": ["he-IL"]},  # Hebrew
    "ku": {"code": "ku", "voices": ["ku-IQ", "ku-TR"]},  # Kurdish variants
    "ps": {"code": "ps", "voices": ["ps-AF"]},  # Pashto
    
    # Southeast Asian with voice options
    "id": {"code": "id", "voices": ["id-ID"]},  # Indonesian
    "ms": {"code": "ms", "voices": ["ms-MY", "ms-BN"]},  # Malay variants
    "th": {"code": "th", "voices": ["th-TH"]},  # Thai
    "vi": {"code": "vi", "voices": ["vi-VN"]},  # Vietnamese
    "km": {"code": "km", "voices": ["km-KH"]},  # Khmer
    "lo": {"code": "lo", "voices": ["lo-LA"]},  # Lao
    "my": {"code": "my", "voices": ["my-MM"]},  # Burmese
    
    # African with voice options
    "af": {"code": "af", "voices": ["af-ZA"]},  # Afrikaans
    "sw": {"code": "sw", "voices": ["sw-KE", "sw-TZ"]},  # Swahili variants
    "yo": {"code": "yo", "voices": ["yo-NG"]},  # Yoruba
    "zu": {"code": "zu", "voices": ["zu-ZA"]},  # Zulu
    "xh": {"code": "xh", "voices": ["xh-ZA"]},  # Xhosa
    "st": {"code": "st", "voices": ["st-ZA", "st-LS"]},  # Sesotho variants
    "sn": {"code": "sn", "voices": ["sn-ZW"]},  # Shona
    "ny": {"code": "ny", "voices": ["ny-MW"]},  # Chichewa
    "rw": {"code": "rw", "voices": ["rw-RW"]},  # Kinyarwanda
    "so": {"code": "so", "voices": ["so-SO", "so-DJ", "so-ET", "so-KE"]},  # Somali variants
    "am": {"code": "am", "voices": ["am-ET"]},  # Amharic
    "ha": {"code": "ha", "voices": ["ha-NG", "ha-GH"]},  # Hausa variants
    "ig": {"code": "ig", "voices": ["ig-NG"]},  # Igbo
    "mg": {"code": "mg", "voices": ["mg-MG"]},  # Malagasy
    
    # Other Languages with voice options
    "ka": {"code": "ka", "voices": ["ka-GE"]},  # Georgian
    "hy": {"code": "hy", "voices": ["hy-AM"]},  # Armenian
    "uz": {"code": "uz", "voices": ["uz-UZ"]},  # Uzbek
    "kk": {"code": "kk", "voices": ["kk-KZ"]},  # Kazakh
    "ky": {"code": "ky", "voices": ["ky-KG"]},  # Kyrgyz
    "tg": {"code": "tg", "voices": ["tg-TJ"]},  # Tajik
    "tk": {"code": "tk", "voices": ["tk-TM"]},  # Turkmen
    "mn": {"code": "mn", "voices": ["mn-MN"]},  # Mongolian
    "bo": {"code": "bo", "voices": ["bo-CN"]},  # Tibetan
    "ti": {"code": "ti", "voices": ["ti-ET", "ti-ER"]},  # Tigrinya variants
    "om": {"code": "om", "voices": ["om-ET", "om-KE"]},  # Oromo variants
    "cy": {"code": "cy", "voices": ["cy-GB"]},  # Welsh
    "ga": {"code": "ga", "voices": ["ga-IE"]},  # Irish
    "mt": {"code": "mt", "voices": ["mt-MT"]},  # Maltese
    
    # Additional Languages
    "lv": {"code": "lv", "voices": ["lv-LV"]},  # Latvian
    "lt": {"code": "lt", "voices": ["lt-LT"]},  # Lithuanian
    "et": {"code": "et", "voices": ["et-EE"]},  # Estonian
    "sl": {"code": "sl", "voices": ["sl-SI"]},  # Slovenian
    "mk": {"code": "mk", "voices": ["mk-MK"]},  # Macedonian
    "sq": {"code": "sq", "voices": ["sq-AL", "sq-MK", "sq-XK"]},  # Albanian variants
    "bs": {"code": "bs", "voices": ["bs-BA"]},  # Bosnian
    "cs": {"code": "cs", "voices": ["cs-CZ"]},  # Czech
    "be": {"code": "be", "voices": ["be-BY"]},  # Belarusian
    "az": {"code": "az", "voices": ["az-AZ"]},  # Azerbaijani
    "uz": {"code": "uz", "voices": ["uz-UZ"]},  # Uzbek
    "tt": {"code": "tt", "voices": ["tt-RU"]},  # Tatar
    "ba": {"code": "ba", "voices": ["ba-RU"]},  # Bashkir
    "cv": {"code": "cv", "voices": ["cv-RU"]},  # Chuvash
    "ce": {"code": "ce", "voices": ["ce-RU"]},  # Chechen
    "os": {"code": "os", "voices": ["os-RU"]},  # Ossetian
    "kbd": {"code": "kbd", "voices": ["kbd-RU"]},  # Kabardian
    "ady": {"code": "ady", "voices": ["ady-RU"]},  # Adyghe
    "inh": {"code": "inh", "voices": ["inh-RU"]},  # Ingush
    "lbe": {"code": "lbe", "voices": ["lbe-RU"]},  # Lak
    "dar": {"code": "dar", "voices": ["dar-RU"]},  # Dargwa
    "lez": {"code": "lez", "voices": ["lez-RU"]},  # Lezgi
    "tab": {"code": "tab", "voices": ["tab-RU"]},  # Tabasaran
    "rut": {"code": "rut", "voices": ["rut-RU"]},  # Rutul
    "agx": {"code": "agx", "voices": ["agx-RU"]},  # Aghul
    "tkr": {"code": "tkr", "voices": ["tkr-RU"]},  # Tsakhur
    "udi": {"code": "udi", "voices": ["udi-RU"]},  # Udi
    "krc": {"code": "krc", "voices": ["krc-RU"]},  # Karachay-Balkar
    "nog": {"code": "nog", "voices": ["nog-RU"]},  # Nogai
    "kum": {"code": "kum", "voices": ["kum-RU"]},  # Kumyk
    "ava": {"code": "ava", "voices": ["ava-RU"]},  # Avar
    "ddo": {"code": "ddo", "voices": ["ddo-RU"]},  # Tsez
    "bua": {"code": "bua", "voices": ["bua-RU"]},  # Buriat
    "xal": {"code": "xal", "voices": ["xal-RU"]},  # Kalmyk
    "tyv": {"code": "tyv", "voices": ["tyv-RU"]},  # Tuvan
    "alt": {"code": "alt", "voices": ["alt-RU"]},  # Southern Altai
    "kjh": {"code": "kjh", "voices": ["kjh-RU"]},  # Khakas
    "cjs": {"code": "cjs", "voices": ["cjs-RU"]},  # Shor
    "ckt": {"code": "ckt", "voices": ["ckt-RU"]},  # Chukchi
    "eve": {"code": "eve", "voices": ["eve-RU"]},  # Even
    "evn": {"code": "evn", "voices": ["evn-RU"]},  # Evenki
    "neg": {"code": "neg", "voices": ["neg-RU"]},  # Negidal
    "oaa": {"code": "oaa", "voices": ["oaa-RU"]},  # Orok
    "ulc": {"code": "ulc", "voices": ["ulc-RU"]},  # Ulch
    "ude": {"code": "ude", "voices": ["ude-RU"]},  # Udege
    "ket": {"code": "ket", "voices": ["ket-RU"]},  # Ket
    "sel": {"code": "sel", "voices": ["sel-RU"]},  # Selkup
    "yrk": {"code": "yrk", "voices": ["yrk-RU"]},  # Nenets
    "mns": {"code": "mns", "voices": ["mns-RU"]},  # Mansi
    "kca": {"code": "kca", "voices": ["kca-RU"]},  # Khanty
    "myv": {"code": "myv", "voices": ["myv-RU"]},  # Erzya
    "mdf": {"code": "mdf", "voices": ["mdf-RU"]},  # Moksha
    "udm": {"code": "udm", "voices": ["udm-RU"]},  # Udmurt
    "koi": {"code": "koi", "voices": ["koi-RU"]},  # Komi-Permyak
    "kpv": {"code": "kpv", "voices": ["kpv-RU"]},  # Komi-Zyrian
    "mhr": {"code": "mhr", "voices": ["mhr-RU"]},  # Eastern Mari
    "mrj": {"code": "mrj", "voices": ["mrj-RU"]},  # Western Mari
    "chm": {"code": "chm", "voices": ["chm-RU"]},  # Mari
}
# Insert after LANG_MAP definition (around line 200-250)
def get_safe_lang_code(lang_code):
    """Return a valid language code for LANG_MAP, or fallback to 'en'."""
    if lang_code in LANG_MAP:
        return lang_code
    st.warning(f"Selected language '{lang_code}' not supported. Using English as fallback.")
    return "en"
# Audio settings
AUDIO_SETTINGS = {
    "speeds": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],  # Available playback speeds
    "default_speed": 1.0,  # Default playback speed
    "volume_range": (0, 100),  # Volume range (0-100)
    "default_volume": 80,  # Default volume
    "pitch_range": (0.5, 2.0),  # Pitch range
    "default_pitch": 1.0,  # Default pitch
}

# Set page config
st.set_page_config(
    page_title="Notes-to-Flashcards AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if "selected_lang_code" not in st.session_state:
        st.session_state.selected_lang_code = "en"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "flashcard_history" not in st.session_state:
        st.session_state.flashcard_history = []
    if "study_stats" not in st.session_state:
        st.session_state.study_stats = {
            "total_cards": 0,
            "cards_studied": 0,
            "correct_answers": 0
        }

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
    return Translator()

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

def generate_flashcards(text: str, num_cards: int, lang_code: str) -> List[Dict[str, str]]:
    """Generate flashcards from input text using OpenAI API"""
    try:
        # Get the appropriate language code for gTTS
        gtts_lang = LANG_MAP.get(lang_code, "en")
        
        # Generate flashcards using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Generate {num_cards} flashcards in {lang_code}. Each flashcard should have a question and answer. Make the content educational and engaging."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the response and create flashcards
        flashcards = []
        content = response.choices[0].message.content
        
        # Split content into individual cards
        cards = content.split("\n\n")
        for card in cards:
            if "Q:" in card and "A:" in card:
                question = card.split("Q:")[1].split("A:")[0].strip()
                answer = card.split("A:")[1].strip()
                
                # Generate audio for question and answer
                audio_question = text_to_audio_base64(question, lang_code)
                audio_answer = text_to_audio_base64(answer, lang_code)
                
                flashcards.append({
                    "question": question,
                    "answer": answer,
                    "audio_question": audio_question,
                    "audio_answer": audio_answer
                })
        
        return flashcards[:num_cards]  # Ensure we only return the requested number of cards
        
    except Exception as e:
        st.error(translate_text(f"Error generating flashcards: {str(e)}", lang_code))
        return []

def text_to_audio_base64(text, lang_code="en", voice=None, speed=1.0):
    """Convert text to audio using gTTS and return as base64 string
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code (e.g., 'en', 'es', 'fr')
        voice (str, optional): Specific voice to use (e.g., 'en-US', 'es-ES')
        speed (float, optional): Speech rate (0.5 to 2.0)
    """
    try:
        # Get language settings
       lang_settings = LANG_MAP[get_safe_lang_code(lang_code)]
       gtts_lang = lang_settings["code"]
        
        # Validate and set voice
       if voice and voice in lang_settings["voices"]:
            gtts_lang = voice
       elif lang_settings["voices"]:
            gtts_lang = lang_settings["voices"][0]  # Use first available voice
        
        # Validate speed
       speed = max(0.5, min(2.0, float(speed)))
        
        # Create a temporary file
       with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech with speed control
       tts = gTTS(text=text, lang=gtts_lang, slow=(speed < 1.0))
       tts.save(temp_filename)
        
        # If speed is not 1.0, adjust the audio speed
       if speed != 1.0:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_filename)
            # Adjust speed by changing frame rate
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)
            })
            audio.export(temp_filename, format="mp3")
        
        # Read the file and convert to base64
       with open(temp_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return audio_base64
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def render_flip_cards(flashcards: List[Dict[str, str]], lang_code: str):
    """Render flashcards with flip animation and audio support"""
    for i, card in enumerate(flashcards):
        # Add unique ID to card
        card['id'] = f"card_{i}"
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                # Front of card (Question)
                st.markdown(f"""
                    <div class="flashcard" onclick="flipCard(this)">
                        <div class="flashcard-inner">
                            <div class="flashcard-front">
                                <h3>{card['question']}</h3>
                            </div>
                            <div class="flashcard-back">
                                <h3>{card['answer']}</h3>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Audio controls for question
                st.markdown(f"**{translate_text('Question Audio', lang_code)}:**")
                card['text'] = card['question']  # Set text for audio generation
                render_audio_controls(card, lang_code)
                
                # Audio controls for answer
                st.markdown(f"**{translate_text('Answer Audio', lang_code)}:**")
                card['text'] = card['answer']  # Update text for audio generation
                render_audio_controls(card, lang_code)
                
                # Add a small gap between cards
                st.markdown("<br>", unsafe_allow_html=True)

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
    st.session_state.study_stats["total_cards"] += len(flashcards)

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
            stats["total_cards"]
        )
    
    with col2:
        matching_win_rate = (
            (stats["correct_answers"] / stats["cards_studied"] * 100)
            if stats["cards_studied"] > 0
            else 0
        )
        st.metric(
            translate_text("Matching Game Win Rate", st.session_state.selected_lang_code),
            f"{matching_win_rate:.1f}%"
        )
    
    with col3:
        practice_pass_rate = (
            (stats["correct_answers"] / stats["cards_studied"] * 100)
            if stats["cards_studied"] > 0
            else 0
        )
        st.metric(
            translate_text("Practice Test Pass Rate", st.session_state.selected_lang_code),
            f"{practice_pass_rate:.1f}%"
        )

def update_study_stats(game_type, won=False):
    """Update study statistics."""
    if game_type == "matching":
        st.session_state.study_stats["cards_studied"] += 1
        if won:
            st.session_state.study_stats["correct_answers"] += 1
    elif game_type == "practice":
        st.session_state.study_stats["cards_studied"] += 1
        if won:
            st.session_state.study_stats["correct_answers"] += 1

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

def process_chat_input(user_input: str) -> str:
    """Process chat input and return response in the selected language."""
    # Get FAQ dictionary in current language
    faq_dict = get_faq_dict_translated(st.session_state.selected_lang_code)
    chatbot = Chatbot(faq_dict, lang_code=st.session_state.selected_lang_code)
    return chatbot.get_fuzzy_response(user_input)

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
        translate_text("Major Languages", st.session_state.selected_lang_code): [
            "English", "Español", "中文", "हिंदी", "Français", "Deutsch", "日本語", "한국어", "Русский", "Português", "العربية", "Italiano"
        ],
        translate_text("South Asian", st.session_state.selected_lang_code): [
            "বাংলা", "తెలుగు", "தமிழ்", "मराठी", "ગુજરાતી", "ಕನ್ನಡ", "മലയാളം", "ਪੰਜਾਬੀ", "اردو", "नेपाली", "සිංහල"
        ],
        translate_text("European", st.session_state.selected_lang_code): [
            "Nederlands", "Polski", "Svenska", "Dansk", "Suomi", "Norsk", "Ελληνικά", "Magyar", "Română", "Slovenčina", "Українська", "Български",
            "Hrvatski", "Српски", "Català", "Euskara", "Galego", "Íslenska"
        ],
        translate_text("Middle Eastern", st.session_state.selected_lang_code): [
            "فارسی", "Türkçe", "עברית", "کوردی", "پښتو"
        ],
        translate_text("Southeast Asian", st.session_state.selected_lang_code): [
            "Bahasa Indonesia", "Bahasa Melayu", "ไทย", "Tiếng Việt", "ខ្មែរ", "ລາວ", "မြန်မာ"
        ],
        translate_text("African", st.session_state.selected_lang_code): [
            "Afrikaans", "Kiswahili", "Yorùbá", "isiZulu", "isiXhosa", "Sesotho", "chiShona", "Chichewa", "Kinyarwanda", "Soomaali", "አማርኛ", "Hausa",
            "Igbo", "Malagasy"
        ],
        translate_text("Other Languages", st.session_state.selected_lang_code): [
            "ქართული", "Հայերեն", "O'zbek", "Қазақ", "Кыргыз", "Тоҷикӣ", "Türkmen", "Монгол", "བོད་སྐད་", "ትግርኛ", "Afaan Oromoo",
            "Cymraeg", "Gaeilge", "Malti"
        ]
    }
    
    # Create tabs for language groups
    lang_tabs = st.sidebar.tabs(list(language_groups.keys()))
    
    # Track if language was changed
    language_changed = False
    
    # Language code mapping
    LANGUAGE_CODES = {
        # Major Languages
        "English": "en", "Español": "es", "中文": "zh", "हिंदी": "hi", "Français": "fr", "Deutsch": "de",
        "日本語": "ja", "한국어": "ko", "Русский": "ru", "Português": "pt", "العربية": "ar", "Italiano": "it",
        
        # South Asian
        "বাংলা": "bn", "తెలుగు": "te", "தமிழ்": "ta", "मराठी": "mr", "ગુજરાતી": "gu", "ಕನ್ನಡ": "kn",
        "മലയാളം": "ml", "ਪੰਜਾਬੀ": "pa", "اردو": "ur", "नेपाली": "ne", "සිංහල": "si",
        
        # European
        "Nederlands": "nl", "Polski": "pl", "Svenska": "sv", "Dansk": "da", "Suomi": "fi", "Norsk": "no",
        "Ελληνικά": "el", "Magyar": "hu", "Română": "ro", "Slovenčina": "sk", "Українська": "uk", "Български": "bg",
        "Hrvatski": "hr", "Српски": "sr", "Català": "ca", "Euskara": "eu", "Galego": "gl", "Íslenska": "is",
        
        # Middle Eastern
        "فارسی": "fa", "Türkçe": "tr", "עברית": "he", "کوردی": "ku", "پښتو": "ps",
        
        # Southeast Asian
        "Bahasa Indonesia": "id", "Bahasa Melayu": "ms", "ไทย": "th", "Tiếng Việt": "vi", "ខ្មែរ": "km",
        "ລາວ": "lo", "မြန်မာ": "my",
        
        # African
        "Afrikaans": "af", "Kiswahili": "sw", "Yorùbá": "yo", "isiZulu": "zu", "isiXhosa": "xh", "Sesotho": "st",
        "chiShona": "sn", "Chichewa": "ny", "Kinyarwanda": "rw", "Soomaali": "so", "አማርኛ": "am", "Hausa": "ha",
        "Igbo": "ig", "Malagasy": "mg",
        
        # Other Languages
        "ქართული": "ka", "Հայերեն": "hy", "O'zbek": "uz", "Қазақ": "kk", "Кыргыз": "ky", "Тоҷикӣ": "tg",
        "Türkmen": "tk", "Монгол": "mn", "བོད་སྐད་": "bo", "ትግርኛ": "ti", "Afaan Oromoo": "om",
        "Cymraeg": "cy", "Gaeilge": "ga", "Malti": "mt"
    }
    
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
                        type="primary" if LANGUAGE_CODES[lang] == st.session_state.selected_lang_code else "secondary"
                    ):
                       safe_lang = get_safe_lang_code(LANGUAGE_CODES[lang])
st.session_state.selected_lang_code = safe_lang
                        language_changed = True
    
    # Show current language
    current_lang = next((k for k, v in LANGUAGE_CODES.items() if v == st.session_state.selected_lang_code), "English")
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

def generate_flashcards_from_ocr(text: str, max_cards: int = 5) -> List[Dict[str, str]]:
    """Generate flashcards from OCR text using simple Q&A splitting."""
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 0]
    flashcards = []
    
    for line in lines:
        # Basic Q&A split: using colon or dash as delimiters
        if ":" in line:
            parts = line.split(":", 1)
            question, answer = parts[0].strip(), parts[1].strip()
            flashcards.append({
                "question": question,
                "answer": answer,
                "explanation": answer,  # Use answer as explanation for OCR cards
                "difficulty": "medium",
                "type": "definition"
            })
        elif "-" in line:
            parts = line.split("-", 1)
            question, answer = parts[0].strip(), parts[1].strip()
            flashcards.append({
                "question": question,
                "answer": answer,
                "explanation": answer,  # Use answer as explanation for OCR cards
                "difficulty": "medium",
                "type": "definition"
            })
        
        if len(flashcards) >= max_cards:
            break
    
    return flashcards

def render_chatbot(faq_dict: Dict[str, str]):
    """Render the chatbot interface using Streamlit's chat components."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        chatbot = Chatbot(faq_dict, 
        get_safe_lang_code(st.session_state.selected_lang_code))

    st.markdown("### 💬 " + translate_text("Need help? Ask me anything!", st.session_state.selected_lang_code))

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(translate_text(message["content"], st.session_state.selected_lang_code))

    # Chat input
    if prompt := st.chat_input(translate_text("Ask a question...", st.session_state.selected_lang_code)):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get response from chatbot
        response, related_questions = chatbot.get_fuzzy_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display the messages
        with st.chat_message("user"):
            st.write(translate_text(prompt, st.session_state.selected_lang_code))
        with st.chat_message("assistant"):
            st.write(translate_text(response, st.session_state.selected_lang_code))
            
            # Display related questions if available
            if related_questions:
                st.write(translate_text("Related questions you might want to ask:", st.session_state.selected_lang_code))
                for question in related_questions:
                    if st.button(question, key=f"related_{question}"):
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.experimental_rerun()

    # Clear chat history button
    if st.button(translate_text("Clear Chat History", st.session_state.selected_lang_code)):
        st.session_state.chat_history = []
        st.experimental_rerun()

import streamlit as st
import yagmail

EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

def send_email(subject, body, to=None):
    if to is None:
        to = EMAIL_USER
    yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
    yag.send(to=to, subject=subject, contents=body)

def render_feedback_form():
    """Render the feedback form in the sidebar (the ONLY feedback form)."""
    with st.sidebar.expander("💬 Feedback / Suggestion Box"):
        name = st.text_input("Your Name", placeholder="Optional")
        feedback_type = st.radio(
            "Feedback Type",
            ["Suggestion", "Bug Report", "Feature Request", "Language Support", "Other"]
        )
        suggestion = st.text_area(
            "Your Feedback / Suggestion",
            placeholder="Please describe your feedback in detail...",
            height=150
        )
        show_contact = st.checkbox("Add contact information (optional)")
        email = ""
        notify_me = False
        if show_contact:
            email = st.text_input("Your email:", placeholder="We'll only use this to follow up on your feedback")
            notify_me = st.checkbox("I'd like to be notified when this is addressed", key="notify_me")
        rating = st.slider(
            "How would you rate your experience?",
            min_value=1,
            max_value=5,
            value=5,
            key="sidebar_app_rating",
            help="1 = Poor, 5 = Excellent"
        )

        if st.button("Submit Feedback", type="primary", key="sidebar_submit_feedback"):
            if suggestion:
                subject = "New Feedback from Notes-to-Flashcards AI"
                body = f"Name: {name}\nFeedback Type: {feedback_type}\nFeedback/Suggestion: {suggestion}\nRating: {rating}"
                if show_contact and email:
                    body += f"\nContact Email: {email}\nNotify: {notify_me}"
                try:
                    send_email(subject, body)
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"Failed to send feedback email: {e}")
                st.experimental_rerun()
            else:
                st.error("Please provide your feedback before submitting.")

def render_image_input():
    """Render the image input section with file upload and camera options."""
    st.write(translate_text("Upload an image of your notes or take a photo:", st.session_state.selected_lang_code))
    
    # Create tabs for upload and camera
    upload_tab, camera_tab = st.tabs([
        translate_text("📤 Upload Image", st.session_state.selected_lang_code),
        translate_text("📷 Take Photo", st.session_state.selected_lang_code)
    ])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            translate_text("Choose an image file (PNG, JPG, JPEG):", st.session_state.selected_lang_code),
            type=["png", "jpg", "jpeg"],
            help=translate_text("Supported formats: PNG, JPG, JPEG", st.session_state.selected_lang_code)
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            try:
                image = Image.open(uploaded_file)
                extracted_text = pytesseract.image_to_string(image)
                if extracted_text.strip():
                    st.success(translate_text("Text extracted successfully!", st.session_state.selected_lang_code))
                    st.text_area(
                        translate_text("Extracted Text:", st.session_state.selected_lang_code),
                        value=extracted_text,
                        height=200,
                        key="extracted_text_upload"
                    )
                    return extracted_text
                else:
                    st.error(translate_text("No text could be extracted from the image. Please try another image.", st.session_state.selected_lang_code))
            except Exception as e:
                st.error(translate_text(f"Error processing image: {str(e)}", st.session_state.selected_lang_code))
    
    with camera_tab:
        st.write(translate_text("Take a photo of your notes using your device's camera:", st.session_state.selected_lang_code))
        camera_image = st.camera_input(translate_text("Take a picture", st.session_state.selected_lang_code))
        
        if camera_image is not None:
            # Process camera image
            try:
                image = Image.open(camera_image)
                extracted_text = pytesseract.image_to_string(image)
                if extracted_text.strip():
                    st.success(translate_text("Text extracted successfully!", st.session_state.selected_lang_code))
                    st.text_area(
                        translate_text("Extracted Text:", st.session_state.selected_lang_code),
                        value=extracted_text,
                        height=200,
                        key="extracted_text_camera"
                    )
                    return extracted_text
                else:
                    st.error(translate_text("No text could be extracted from the photo. Please try again.", st.session_state.selected_lang_code))
            except Exception as e:
                st.error(translate_text(f"Error processing photo: {str(e)}", st.session_state.selected_lang_code))
    
    return None

def render_audio_controls(card, lang_code):
    """Render audio controls with voice selection and speed control"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Voice selection
        lang_settings = LANG_MAP.get(lang_code, {"voices": ["en-US"]})
        voices = lang_settings["voices"]
        selected_voice = st.selectbox(
            translate_text("Voice", lang_code),
            options=voices,
            key=f"voice_{card['id']}"
        )
    
    with col2:
        # Speed control
        speed = st.slider(
            translate_text("Speed", lang_code),
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.25,
            key=f"speed_{card['id']}"
        )
    
    with col3:
        # Play button
        if st.button("▶️", key=f"play_{card['id']}"):
            # Generate audio with selected settings
            audio_data = text_to_audio_base64(
                card['text'],
                lang_code,
                voice=selected_voice,
                speed=speed
            )
            if audio_data:
                st.audio(audio_data, format="audio/mp3")

def main():
    initialize_session_state()
    
    # Render language selector in sidebar
    render_language_selector()
    
    # Render feedback form in sidebar
    render_feedback_form()
    
    # Main content
    st.title("📘 Notes-to-Flashcards AI")
    
    # Language-specific welcome message
    st.info(translate_text(
        "Welcome! This app supports multiple languages. Choose your preferred language from the sidebar. "
        "All content, including flashcards and chatbot responses, will be displayed in your selected language.",
        st.session_state.selected_lang_code
    ))

    # Create tabs for different input methods and feedback
    input_tab1, input_tab2, feedback_tab, chat_tab = st.tabs([
        translate_text("📝 Text Input", st.session_state.selected_lang_code),
        translate_text("🖼️ Image Upload", st.session_state.selected_lang_code),
        translate_text("💡 Feedback", st.session_state.selected_lang_code),
        translate_text("💬 Chat", st.session_state.selected_lang_code)
    ])

    with input_tab1:
        # Text input area
text_input = st.text_area(
    translate_text("Enter or paste your notes here:", st.session_state.selected_lang_code),
    height=200,
    key="text_input"
)

# ADD: Number of flashcards selector and warning
num_cards = st.number_input(
    translate_text("How many flashcards do you want?", st.session_state.selected_lang_code),
    min_value=1, max_value=50, value=5, step=1,
    key="num_cards_text"
)
if num_cards > 30:
    st.warning(translate_text("Generating more than 30 flashcards may take longer.", st.session_state.selected_lang_code))

if text_input:
    if st.button(translate_text("Generate Flashcards", st.session_state.selected_lang_code), type="primary"):
        with st.spinner(translate_text("Generating flashcards...", st.session_state.selected_lang_code)):
            flashcards = generate_flashcards(text_input, num_cards, st.session_state.selected_lang_code)
                    else:
                        st.error(translate_text("Failed to generate flashcards. Please try again.", st.session_state.selected_lang_code))

 with input_tab2:
    extracted_text = render_image_input()

    # ADD: Number selector and warning
    num_cards_img = st.number_input(
        translate_text("How many flashcards do you want?", st.session_state.selected_lang_code),
        min_value=1, max_value=50, value=5, step=1,
        key="num_cards_img"
    )
    if num_cards_img > 30:
        st.warning(translate_text("Generating more than 30 flashcards may take longer.", st.session_state.selected_lang_code))

    if extracted_text:
        if st.button(translate_text("Generate Flashcards", st.session_state.selected_lang_code), type="primary"):
            with st.spinner(translate_text("Generating flashcards...", st.session_state.selected_lang_code)):
                flashcards = generate_flashcards(extracted_text, num_cards_img, st.session_state.selected_lang_code)

                    if flashcards:
                        st.success(translate_text(f"Generated {len(flashcards)} flashcards!", st.session_state.selected_lang_code))
                        save_flashcards_to_history(flashcards)
                        render_flip_cards(flashcards, st.session_state.selected_lang_code)
                    else:
                        st.error(translate_text("Failed to generate flashcards. Please try again.", st.session_state.selected_lang_code))

    with feedback_tab:
        st.info("To provide feedback, please use the feedback form in the sidebar.")

    with chat_tab:
        render_chatbot(FAQ_DICT_EN)

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

from deep_translator import GoogleTranslator
import streamlit as st

# Initialize translator with caching
@st.cache_resource
def get_translator():
    """Get a cached instance of GoogleTranslator."""
    return GoogleTranslator(source='auto', target='en')  # Default to English, will be changed per translation

# Language codes and names
LANGUAGES = {
    "English": "en",
    "Español": "es",
    "हिंदी": "hi",
    "Français": "fr",
    "Deutsch": "de",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
    "Русский": "ru",
    "Português": "pt",
    "العربية": "ar",
    "Italiano": "it",
    "বাংলা": "bn",
    "اردو": "ur",
    "Türkçe": "tr",
    "Tiếng Việt": "vi",
    "Polski": "pl",
    "Nederlands": "nl",
    "Bahasa Indonesia": "id",
    "Svenska": "sv",
    "తెలుగు": "te"
}

def translate_text(text: str, dest: str = 'en') -> str:
    """Translate text to the specified language using Google Translate.
    
    Args:
        text (str): The text to translate
        dest (str): Destination language code (default: 'en')
        
    Returns:
        str: Translated text, or original text if translation fails
    """
    if not text or dest == 'en':  # No translation needed for empty text or English
        return text
        
    try:
        translator = get_translator()
        translator.target = dest  # Update target language
        result = translator.translate(text)
        return result
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original text.")
        return text

def get_translation(key: str, lang_code: str = "en") -> str:
    """Get translation for a key in the specified language."""
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(key, key)

# Dictionary of translations for UI elements
TRANSLATIONS = {
    "en": {
        "title": "🧠 Notes to Flashcards AI",
        "chat_header": "💬 Interactive Chat",
        "chat_input_placeholder": "Type your question here...",
        "send_button": "Send",
        "clear_chat": "Clear Chat History",
        "upload_header": "📤 Upload or Input Your Notes",
        "upload_file": "Upload a file (PDF, DOCX, TXT)",
        "upload_help": "Supported formats: PDF, DOCX, TXT",
        "type_notes": "Enter your notes here",
        "type_placeholder": "Type or paste your notes here...",
        "upload_image": "Upload an image of your notes",
        "image_help": "Supported formats: PNG, JPG, JPEG",
        "take_photo": "Take a photo of your notes",
        "notes_display": "📄 Your Notes",
        "generate_button": "Summarize and Generate Flashcards",
        "summary_generated": "✅ Summary Generated:",
        "flashcards_ready": "✅ Flashcards Ready:",
        "upload_success": "✅ Notes extracted from uploaded file.",
        "type_success": "✅ Notes taken from typed input.",
        "image_success": "✅ Notes extracted from uploaded image.",
        "photo_success": "✅ Notes extracted from photo.",
        "please_input": "👆 Please choose one of the input methods above to get started.",
        "thinking": "Thinking...",
        "summarizing": "Summarizing...",
        "generating_flashcards": "Generating flashcards...",
        "footer": "Made with ❤️ using Streamlit, BART, and FLAN-T5",
        "tab_upload": "📄 Upload File",
        "tab_type": "✍️ Type/Paste",
        "tab_image": "📸 Upload Image",
        "tab_photo": "📱 Take Photo",
        "language_settings": "🌐 Language Settings",
        "select_language": "Choose your language",
        "current_language": "Current Language:",
        "language_changed": "Language changed! The interface will update.",
        "contact_info": "Contact Information (Optional)",
        "your_email": "Your email:",
        "notify_me": "I'd like to be notified when this is addressed",
        "feedback_title": "💬 Feedback / Suggestion Box",
        "your_name": "Your Name",
        "optional": "Optional",
        "feedback_type": "Feedback Type",
        "suggestion": "Suggestion",
        "bug_report": "Bug Report",
        "feature_request": "Feature Request",
        "language_support": "Language Support",
        "other": "Other",
        "your_feedback": "Your Feedback / Suggestion",
        "feedback_placeholder": "Please describe your feedback in detail...",
        "submit_feedback": "Submit Feedback",
        "thank_you_feedback": "Thank you for your feedback! We'll review it and get back to you if needed.",
        "please_provide_feedback": "Please provide your feedback before submitting.",
        "experience_rating": "How would you rate your experience?",
        "rating_help": "1 = Poor, 5 = Excellent"
    },
    "es": {
        "title": "🧠 Notas a Tarjetas de Memoria IA",
        "chat_header": "💬 Chat Interactivo",
        "chat_input_placeholder": "Escribe tu pregunta aquí...",
        "send_button": "Enviar",
        "clear_chat": "Borrar Historial de Chat",
        "upload_header": "📤 Subir o Ingresar Notas",
        "upload_file": "Subir archivo (PDF, DOCX, TXT)",
        "upload_help": "Formatos soportados: PDF, DOCX, TXT",
        "type_notes": "Ingresa tus notas aquí",
        "type_placeholder": "Escribe o pega tus notas aquí...",
        "upload_image": "Subir imagen de tus notas",
        "image_help": "Formatos soportados: PNG, JPG, JPEG",
        "take_photo": "Tomar foto de tus notas",
        "notes_display": "📄 Tus Notas",
        "generate_button": "Resumir y Generar Tarjetas",
        "summary_generated": "✅ Resumen Generado:",
        "flashcards_ready": "✅ Tarjetas Listas:",
        "upload_success": "✅ Notas extraídas del archivo subido.",
        "type_success": "✅ Notas tomadas de la entrada escrita.",
        "image_success": "✅ Notas extraídas de la imagen subida.",
        "photo_success": "✅ Notas extraídas de la foto.",
        "please_input": "👆 Por favor, elige uno de los métodos de entrada anteriores para comenzar.",
        "thinking": "Pensando...",
        "summarizing": "Resumiendo...",
        "generating_flashcards": "Generando tarjetas...",
        "footer": "Hecho con ❤️ usando Streamlit, BART y FLAN-T5",
        "tab_upload": "📄 Subir Archivo",
        "tab_type": "✍️ Escribir/Pegar",
        "tab_image": "📸 Subir Imagen",
        "tab_photo": "📱 Tomar Foto",
        "language_settings": "🌐 Configuración de Idioma",
        "select_language": "Elige tu idioma"
    },
    # Add more languages as needed...
} 
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple
import streamlit as st

class Chatbot:
    def __init__(self, faq_dict: Dict[str, str], lang_code: str = "en"):
        # Convert all keys to lowercase for case-insensitive matching
        self.faq_dict = {k.lower(): v for k, v in faq_dict.items()}
        self.questions = list(faq_dict.keys())
        self.lang_code = lang_code
        
        # Initialize conversation history if not exists
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # Define question categories
        self.categories = {
            "general": [
                "how do i use this app?",
                "what languages are supported?",
                "is this app free?",
                "how do i contact support?"
            ],
            "flashcards": [
                "how do i create flashcards?",
                "can i edit my flashcards?",
                "how do i study with flashcards?",
                "can i share my flashcards?"
            ],
            "audio": [
                "how does audio playback work?",
                "which languages support audio?",
                "can i adjust the audio speed?",
                "is audio available offline?"
            ],
            "language": [
                "how do i change the language?",
                "can i add new languages?",
                "are all features available in my language?",
                "how accurate are the translations?"
            ]
        }
        
        # Define related questions for common topics
        self.related_questions = {
            "flashcard": [
                "How do I create flashcards?",
                "Can I edit my flashcards?",
                "How do I study with flashcards?",
                "Can I share my flashcards?",
                "How do I delete flashcards?"
            ],
            "language": [
                "How do I change the language?",
                "Which languages are supported?",
                "Can I add new languages?",
                "How do I report translation issues?"
            ],
            "audio": [
                "How does audio playback work?",
                "Which languages support audio?",
                "Can I adjust the audio speed?",
                "Is audio available offline?"
            ]
        }

    def get_response(self, question: str) -> str:
        """Get response for a question using exact matching."""
        return self.faq_dict.get(
            question.lower().strip(),
            "I'm sorry, I don't know the answer to that. Try asking something else!"
        )

    def __call__(self, question: str) -> str:
        """Make the chatbot instance callable like a function."""
        return self.get_response(question)

    def get_fuzzy_response(self, prompt: str) -> Tuple[str, List[str]]:
        """Get response using fuzzy matching with context awareness."""
        # Add to conversation history
        st.session_state.conversation_history.append({"role": "user", "text": prompt})
        
        # Try exact match first
        response = self.get_response(prompt)
        if response != "I'm sorry, I don't know the answer to that. Try asking something else!":
            return response, self._get_related_questions(prompt)
        
        # Try fuzzy match
        matches = get_close_matches(prompt.lower(), [q.lower() for q in self.questions], n=1, cutoff=0.6)
        if matches:
            return self.faq_dict[matches[0].lower()], self._get_related_questions(matches[0])
        
        # Return default response with suggested questions
        default_responses = {
            "en": "I'm sorry, I don't know the answer to that. Here are some questions you might want to ask:",
            "es": "Lo siento, no sé la respuesta a eso. Aquí hay algunas preguntas que podrías querer hacer:",
            "fr": "Désolé, je ne connais pas la réponse à cela. Voici quelques questions que vous pourriez vouloir poser :",
            "de": "Entschuldigung, ich weiß die Antwort darauf nicht. Hier sind einige Fragen, die Sie stellen könnten:",
            "it": "Mi dispiace, non conosco la risposta a questo. Ecco alcune domande che potresti voler fare:",
            "pt": "Desculpe, não sei a resposta para isso. Aqui estão algumas perguntas que você pode querer fazer:",
            "ru": "Извините, я не знаю ответа на этот вопрос. Вот несколько вопросов, которые вы можете задать:",
            "zh": "抱歉，我不知道这个问题的答案。以下是一些您可能想问的问题：",
            "ja": "申し訳ありませんが、その質問の答えはわかりません。以下の質問をしてみてはいかがでしょうか：",
            "ko": "죄송합니다. 그 질문에 대한 답을 모릅니다. 다음 질문을 해보시는 것은 어떨까요?",
            "ar": "عذراً، لا أعرف الإجابة على ذلك. إليك بعض الأسئلة التي قد ترغب في طرحها:",
            "hi": "क्षमा करें, मुझे इसका उत्तर नहीं पता है। यहां कुछ प्रश्न हैं जो आप पूछना चाह सकते हैं:",
            "te": "క్షమించండి, దానికి సమాధానం తెలియదు. మీరు అడగదలచుకున్న కొన్ని ప్రశ్నలు ఇక్కడ ఉన్నాయి:"
        }
        
        # Get suggested questions based on the prompt
        suggested_questions = self._get_suggested_questions(prompt)
        return default_responses.get(self.lang_code, default_responses["en"]), suggested_questions

    def _get_related_questions(self, question: str) -> List[str]:
        """Get related questions based on the current question."""
        question_lower = question.lower()
        related = []
        
        # Check each category for matches
        for category, questions in self.categories.items():
            if any(q in question_lower for q in questions):
                related.extend(questions)
        
        # Add specific related questions based on keywords
        for keyword, questions in self.related_questions.items():
            if keyword in question_lower:
                related.extend(questions)
        
        # Remove duplicates and limit to 5 questions
        return list(dict.fromkeys(related))[:5]

    def _get_suggested_questions(self, prompt: str) -> List[str]:
        """Get suggested questions based on the conversation context."""
        prompt_lower = prompt.lower()
        suggested = []
        
        # Check for keywords in the prompt
        for keyword, questions in self.related_questions.items():
            if keyword in prompt_lower:
                suggested.extend(questions)
        
        # If no specific suggestions, return general questions
        if not suggested:
            suggested = self.categories["general"]
        
        # Remove duplicates and limit to 5 questions
        return list(dict.fromkeys(suggested))[:5]

    def get_quick_help_buttons(self) -> List[Dict[str, str]]:
        """Get a list of quick help buttons for common questions."""
        return [
            {"category": "General", "questions": self.categories["general"]},
            {"category": "Flashcards", "questions": self.categories["flashcards"]},
            {"category": "Audio", "questions": self.categories["audio"]},
            {"category": "Language", "questions": self.categories["language"]}
        ] 
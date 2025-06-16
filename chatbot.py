from difflib import get_close_matches

class Chatbot:
    def __init__(self, faq_dict):
        # Convert all keys to lowercase for case-insensitive matching
        self.faq_dict = {k.lower(): v for k, v in faq_dict.items()}
        self.questions = list(faq_dict.keys())

    def get_response(self, question):
        """Get response for a question using exact matching."""
        return self.faq_dict.get(
            question.lower().strip(),
            "I'm sorry, I don't know the answer to that. Try asking something else!"
        )

    def __call__(self, question):
        """Make the chatbot instance callable like a function."""
        return self.get_response(question)

    def get_fuzzy_response(self, prompt):
        # Try fuzzy match
        matches = get_close_matches(prompt, self.questions, n=1, cutoff=0.6)
        if matches:
            return self.faq_dict[matches[0]]

        return "I'm sorry, I don't know the answer to that. Try asking something else!" 
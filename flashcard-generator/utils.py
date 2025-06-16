import PyPDF2
import docx
import io
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
from PIL import Image
import streamlit as st
from typing import List

# Initialize models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    
    # Load model for question generation
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    return summarizer, tokenizer, model

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on its type."""
    if uploaded_file is None:
        return None
        
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            return uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file_type == 'docx':
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return None

def extract_text_from_image(image_file):
    """Extract text from an image using OCR."""
    try:
        if isinstance(image_file, Image.Image):
            img = image_file
        else:
            img = Image.open(image_file)
        return pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None

def generate_summary(text, summarizer=None):
    """Generate a summary of the input text."""
    if summarizer is None:
        summarizer, _, _ = load_models()
    
    # Split text into chunks if it's too long
    max_chunk_length = 1000
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def generate_question(text, tokenizer, model):
    """Generate a question from the input text."""
    prompt = f"Generate a question about this text: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def generate_flashcards(text, num_flashcards=5, tokenizer=None, model=None):
    """Generate flashcards from the input text."""
    if tokenizer is None or model is None:
        _, tokenizer, model = load_models()
    
    # Generate summary first
    summary = generate_summary(text)
    
    # Generate questions
    questions_prompt = f"Generate {num_flashcards} questions from this text: {summary}"
    questions_response = model.generate(
        tokenizer(questions_prompt, return_tensors="pt", max_length=512, truncation=True)["input_ids"],
        max_length=200,
        num_beams=4,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    questions_text = tokenizer.decode(questions_response[0], skip_special_tokens=True)
    
    # Process questions and answers
    flashcard_pairs = []
    
    # Add main idea as first card
    flashcard_pairs.append({
        "question": "What is the main idea of the text?",
        "answer": summary
    })
    
    # Process generated questions
    for line in questions_text.splitlines():
        q = line.strip()
        if not q:
            continue
            
        # Remove numbering if present (e.g., "1. " or "1) ")
        if q[0].isdigit() and len(q) > 2 and q[1] in [".", ")"]:
            q = q[2:].strip()
            
        if q:
            # Generate answer for each question
            answer_prompt = f"Answer this question: {q}"
            answer_response = model.generate(
                tokenizer(answer_prompt, return_tensors="pt", max_length=512, truncation=True)["input_ids"],
                max_length=200,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=2
            )
            answer = tokenizer.decode(answer_response[0], skip_special_tokens=True).strip()
            
            flashcard_pairs.append({
                "question": q,
                "answer": answer
            })
    
    return flashcard_pairs

def chunk_text(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of specified size.
    
    Args:
        text (str): The text to split into chunks
        max_chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
        
    # Split text into sentences to avoid breaking mid-sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed the chunk size
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_text = ' '.join(current_chunk[-overlap:]) if overlap > 0 else ''
            current_chunk = [overlap_text] if overlap_text else []
            current_size = len(overlap_text)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 
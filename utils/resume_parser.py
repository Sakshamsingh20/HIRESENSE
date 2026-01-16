import PyPDF2

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
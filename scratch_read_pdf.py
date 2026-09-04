import os
import sys

def try_extract_pypdf():
    try:
        import pypdf
        print("Using pypdf")
        reader = pypdf.PdfReader("ROADMAP.pdf")
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            text += page.extract_text() or ""
            text += "\n\n"
        return text
    except ImportError:
        return None

def try_extract_pypdf2():
    try:
        import PyPDF2
        print("Using PyPDF2")
        reader = PyPDF2.PdfReader("ROADMAP.pdf")
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            text += page.extract_text() or ""
            text += "\n\n"
        return text
    except ImportError:
        return None

def try_extract_pdfplumber():
    try:
        import pdfplumber
        print("Using pdfplumber")
        text = ""
        with pdfplumber.open("ROADMAP.pdf") as pdf:
            for i, page in enumerate(pdf.pages):
                text += f"--- Page {i+1} ---\n"
                text += page.extract_text() or ""
                text += "\n\n"
        return text
    except ImportError:
        return None

def try_extract_fitz():
    try:
        import fitz  # PyMuPDF
        print("Using fitz")
        text = ""
        doc = fitz.open("ROADMAP.pdf")
        for i, page in enumerate(doc):
            text += f"--- Page {i+1} ---\n"
            text += page.get_text()
            text += "\n\n"
        return text
    except ImportError:
        return None

def main():
    text = try_extract_pypdf()
    if text is None:
        text = try_extract_pypdf2()
    if text is None:
        text = try_extract_pdfplumber()
    if text is None:
        text = try_extract_fitz()
    
    if text is None:
        print("No PDF reader library found in standard system.")
        sys.exit(1)
        
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/roadmap_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully wrote roadmap text to scratch/roadmap_text.txt")

if __name__ == "__main__":
    main()

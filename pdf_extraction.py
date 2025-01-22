import fitz

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf = fitz.open(pdf_path)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text += page.get_text() + "\n"
    pdf.close()

    return text
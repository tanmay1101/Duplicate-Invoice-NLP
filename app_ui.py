import pandas as pd
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pdfplumber

# GitHub URL for master_df.csv
master_df_url = r'https://raw.githubusercontent.com/tanmay1101/Duplicate-Invoice-NLP/main/master_df.csv'

def preprocess_image(image):
    image = image.convert('L')
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    return image

def extract_text_from_image(image):
    return pytesseract.image_to_string(image, config='--oem 3 --psm 6')

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def extract_invoice_info(text):
    invoice_number_pattern = r'Invoice no[:\s]*([A-Z0-9]+)'
    date_pattern = r'Date of issue[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})'
    invoice_number_match = re.search(invoice_number_pattern, text, re.IGNORECASE)
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    invoice_number = invoice_number_match.group(1) if invoice_number_match else 'Not found'
    date = date_match.group(1) if date_match else 'Not found'
    invoice_summary = re.sub(f'{invoice_number_pattern}|{date_pattern}', '', text, flags=re.IGNORECASE).strip()
    return invoice_number, date, invoice_summary

def calculate_similarity(df1, df2):
    df1['Combined'] = df1.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df2['Combined'] = df2.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    vectorizer = TfidfVectorizer()
    combined_texts = df1['Combined'].tolist() + df2['Combined'].tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    similarities = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])
    similarity_percentages = [max(sim) * 100 for sim in similarities]
    df1['Similarity'] = similarity_percentages
    
    # Find matched rows in master_df with similarity above a threshold
    threshold = 70
    matched_indices = [i for i, score in enumerate(similarity_percentages) if score > threshold]
    matched_rows = df2.iloc[matched_indices] if matched_indices else pd.DataFrame()
    
    return df1, matched_rows

def is_invoice_text(text):
    # Define simple invoice detection criteria
    invoice_keywords = ['invoice', 'invoice no', 'invoice number', 'date of issue', 'Invoice','INVOICE']
    return any(keyword.lower() in text.lower() for keyword in invoice_keywords)

def process_pdf(pdf_path, master_df):
    extracted_text = extract_text_from_pdf(pdf_path)
    if is_invoice_text(extracted_text):
        if extracted_text.strip():
            data = []
            invoice_number, date, invoice_summary = extract_invoice_info(extracted_text)
            data.append([invoice_number, date, invoice_summary])
            df = pd.DataFrame(data, columns=['Invoice no', 'Date of issue', 'Invoice Summary'])
        else:
            images = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Gen-AI\Invoice Fraud\poppler-24.07.0\Library\bin')
            data = []
            for i, image in enumerate(images):
                processed_image = preprocess_image(image)
                text = extract_text_from_image(processed_image)
                invoice_number, date, invoice_summary = extract_invoice_info(text)
                data.append([invoice_number, date, invoice_summary])
            df = pd.DataFrame(data, columns=['Invoice no', 'Date of issue', 'Invoice Summary'])
        df, matched_rows = calculate_similarity(df, master_df)
        return df, matched_rows, True
    else:
        return pd.DataFrame(), pd.DataFrame(), False

# Streamlit App
st.set_page_config(page_title="Duplicate Invoice Analyser", page_icon=":memo:", layout="wide")

st.markdown("""
    <style>
        body {
            background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ggJPdXD_9ZML8Yyky5usLh9Dc3iNIaZNMQ&s');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }
        .title {
            color: green;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
            padding: 0.5em 0;
        }
        .subtitle {
            color: white;
            text-align: center;
            font-size: 1.5em;
            font-weight: normal;
            margin: 0;
            padding: 0em 0;
        }
        .stFileUploader>div>div {
            background-color: black;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 1em;
        }
        .stFileUploader>div>div>input {
            border: none;
            color: white;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stDataFrame>div>div>div>div {
            border: 1px solid #4CAF50;
        }
        .stApp {
            background-color: rgba(0, 0, 0, 0.5);
        }
        .stMarkdown p, .stWrite {
            color: white;
        }
        .stError {
            background-color: black;
            color: red;
            padding: 1em;
            border-radius: 5px;
            font-size: 1.2em;
        }
        .stSuccess {
            background-color: black;
            color: green;
            padding: 1em;
            border-radius: 5px;
            font-size: 1.2em;
        }
        .file-name {
            color: green;
            font-size: 1.2em;
            text-align: right;
            margin-top: 1em;
            display: block;
        }
        .stDataFrame {
            background-color: black;
        }
        .stDataFrame table {
            color: white;
        }
        .stDataFrame th {
            background-color: #333;
            color: white;
        }
        .stDataFrame td {
            background-color: black;
            color: white;
        }
        .upload-instructions {
            text-align: left;
            font-size: 1.2em;
            color: white;
            margin-bottom: 0em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Duplicate Invoice Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Audit & Assurance</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-instructions">Please Upload Invoice</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Invoice PDF", type="pdf")

if uploaded_file is not None:
    st.markdown(f'<div class="file-name">Uploaded File: {uploaded_file.name}</div>', unsafe_allow_html=True)
    
    with open("uploaded_invoice.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    master_df = pd.read_csv(master_df_url)
    
    result_df, matched_rows, is_invoice = process_pdf("uploaded_invoice.pdf", master_df)
    
    if is_invoice:
        similarity = result_df['Similarity'].iloc[0] if not result_df.empty else 0
        
        st.write("Extracted Invoice Details and Duplicate Check Results:")
        if similarity > 70:
            st.markdown(f"<div class='stError'>Duplicate invoice detected with {similarity:.2f}% similarity.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='stSuccess'>Invoice is correct, no duplicate detected.</div>", unsafe_allow_html=True)
        
        st.dataframe(result_df.style.set_table_styles(
            [{'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
             {'selector': 'tbody td', 'props': [('background-color', 'black'), ('color', 'white')]}]
        ))
        
        if not matched_rows.empty:
            st.write("Matched Rows from Master Data:")
            st.dataframe(matched_rows.style.set_table_styles(
                [{'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
                 {'selector': 'tbody td', 'props': [('background-color', 'black'), ('color', 'white')]}]
            ))
        else:
            st.write("No matching rows found in master data.")
        
        st.download_button(
            label="Download Results as CSV",
            data=result_df.to_csv(index=False),
            file_name='invoice_fraud_check_results.csv',
            mime='text/csv'
        )
    else:
        st.markdown("<div class='stError'>This is not an invoice PDF. Please upload a valid invoice PDF.</div>", unsafe_allow_html=True)

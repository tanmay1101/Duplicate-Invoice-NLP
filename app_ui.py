import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pdfplumber

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale, enhance contrast, and sharpen the image
    image = image.convert('L')  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    return image

def extract_text_from_image(image):
    # Use pytesseract to extract text from the image
    return pytesseract.image_to_string(image, config='--oem 3 --psm 6')

def extract_text_from_pdf(pdf_path):
    # Use pdfplumber to extract text from a text-based PDF
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''  # Extract text or skip if none
    return text

def extract_invoice_info(text):
    # Use regex to find patterns for Invoice Number and Date
    invoice_number_pattern = r'Invoice no[:\s]*([A-Z0-9]+)'  # Adjusted pattern
    date_pattern = r'Date of issue[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})'  # Adjusted pattern

    invoice_number_match = re.search(invoice_number_pattern, text, re.IGNORECASE)
    date_match = re.search(date_pattern, text, re.IGNORECASE)

    invoice_number = invoice_number_match.group(1) if invoice_number_match else 'Not found'
    date = date_match.group(1) if date_match else 'Not found'

    # Extract the rest of the text for the invoice summary
    invoice_summary = re.sub(f'{invoice_number_pattern}|{date_pattern}', '', text, flags=re.IGNORECASE).strip()

    return invoice_number, date, invoice_summary

def calculate_similarity(df1, df2):
    # Combine all columns into a single 'Combined' column for both DataFrames
    df1['Combined'] = df1.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df2['Combined'] = df2.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the Combined columns into vectors
    combined_texts = df1['Combined'].tolist() + df2['Combined'].tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])

    # Flatten and get the maximum similarity for each row in df1
    similarity_percentages = [max(sim) * 100 for sim in similarities]

    # Add similarity percentages to df1
    df1['Similarity'] = similarity_percentages
    return df1

def process_pdf(pdf_path, master_df):
    # Check if the PDF is text-based or image-based by attempting to extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text.strip():  # If text is found, treat as a text-based PDF
        data = []
        invoice_number, date, invoice_summary = extract_invoice_info(extracted_text)
        data.append([invoice_number, date, invoice_summary])
        df = pd.DataFrame(data, columns=['Invoice no', 'Date of issue', 'Invoice Summary'])
    else:  # Otherwise, treat as an image-based PDF
        images = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Gen-AI\Invoice Fraud\poppler-24.07.0\Library\bin')
        data = []

        for i, image in enumerate(images):
            processed_image = preprocess_image(image)
            text = extract_text_from_image(processed_image)
            invoice_number, date, invoice_summary = extract_invoice_info(text)
            data.append([invoice_number, date, invoice_summary])

        df = pd.DataFrame(data, columns=['Invoice no', 'Date of issue', 'Invoice Summary'])

    # Calculate similarity between df and master_df
    df = calculate_similarity(df, master_df)

    return df

# Streamlit App
st.set_page_config(page_title="Duplicate Invoice Analyser", page_icon=":memo:", layout="wide")

# Add custom CSS for full background image, black upload interface, larger font size for file uploader, minimized gap between title and subtitle, custom message styles, file name color, and DataFrame styling
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
            padding: 0.2em 0; /* Reduced padding to minimize gap */
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
            font-size: 18px; /* Increase font size */
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stDataFrame>div>div>div>div {
            border: 1px solid #4CAF50;
        }
        .stApp {
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black background for readability */
        }
        .stMarkdown p, .stWrite {
            color: white; /* Ensure default text messages are white */
        }
        .stError {
            background-color: black; /* Black background for error messages */
            color: red; /* Red text for error messages */
            padding: 1em;
            border-radius: 5px;
            font-size: 1.2em;
        }
        .stSuccess {
            background-color: black; /* Black background for success messages */
            color: green; /* Green text for success messages */
            padding: 1em;
            border-radius: 5px;
            font-size: 1.2em;
        }
        .file-name {
            color: green; /* Green color for file name */
            font-size: 1.2em; /* Slightly larger font size for visibility */
            text-align: right;
            
            display: block;
        }
        .stDataFrame {
            background-color: black; /* Black background for DataFrame */
        }
        .stDataFrame table {
            color: white; /* White text for DataFrame */
        }
        .stDataFrame th {
            background-color: #333; /* Darker background for table headers */
            color: white; /* White text for table headers */
        }
        .stDataFrame td {
            background-color: black; /* Black background for table cells */
            color: white; /* White text for table cells */
        }
        .upload-instructions {
            text-align: left;
            font-size: 1.2em;
            color: white;
            
        }
    </style>
    <div class="title">
        Duplicate Invoice Analyser
    </div>
    <div class="subtitle">
        Audit & Assurance
    </div>
    <div class="upload-instructions">
        Please Upload Invoice
    </div>
    """, unsafe_allow_html=True)

# Upload PDF
uploaded_file = st.file_uploader("Upload Invoice PDF", type="pdf")

if uploaded_file is not None:
    # Display the file name in green
    st.markdown(f'<div class="file-name">Uploaded File: {uploaded_file.name}</div>', unsafe_allow_html=True)
    
    # Save uploaded file temporarily
    with open("uploaded_invoice.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the master DataFrame
    master_df_path = r'C:\Gen-AI\Invoice Fraud\master_df.csv'
    master_df = pd.read_csv(master_df_path)
    
    # Process the uploaded PDF
    result_df = process_pdf("uploaded_invoice.pdf", master_df)
    similarity = result_df['Similarity'].iloc[0]
    
    # Display the results
    st.write("Extracted Invoice Details and Fraud Check Results:")
    if similarity > 70:
        st.markdown(f"<div class='stError'>Duplicate invoice detected with {similarity:.2f}% similarity.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stSuccess'>Invoice is correct, no duplicate detected.</div>", unsafe_allow_html=True)
    
    # Display the DataFrame with custom styling
    st.dataframe(result_df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
         {'selector': 'tbody td', 'props': [('background-color', 'black'), ('color', 'white')]}]
    ))
    
    # Save or download the results
    st.download_button(
        label="Download Results as CSV",
        data=result_df.to_csv(index=False),
        file_name='invoice_fraud_check_results.csv',
        mime='text/csv'
    )

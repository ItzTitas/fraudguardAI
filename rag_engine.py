import pdfplumber
import pandas as pd
import os
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def extract_transactions_from_pdf(pdf_path, groq_api_key=None):
    """
    Extracts transaction table data from the bank statement PDF using a multi-stage approach.
    1. Standard Table Extraction
    2. Heuristic Regex (Row-by-row) detection
    3. AI-Assisted Parsing (Fallback)
    """
    transactions = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # STAGE 1: Standard Table Extraction
                table = page.extract_table()
                if table and len(table) > 1:
                    headers = [str(h).upper() if h else "" for h in table[0]]
                    # Identify if this looks like a transaction table
                    is_tx_table = any(h in ["DATE", "PARTICULARS", "TRANSACTION", "DESCRIPTION", "AMOUNT"] for h in headers)
                    
                    if is_tx_table:
                        df = pd.DataFrame(table[1:], columns=headers)
                        df = df.dropna(how='all')
                        transactions.extend(df.to_dict('records'))
                        continue # Found table, move to next page
                
                # STAGE 2: Heuristic Regex Row Detection (Fallback if table extraction failed)
                text = page.extract_text()
                # If extract_text fails to get meaningful content but words exist, join them
                if (not text or len(text.strip()) < 20):
                    words = page.extract_words()
                    if words:
                        text = " ".join([w['text'] for w in words])

                if text:
                    lines = text.split('\n')
                    # Regex for Date (DD-MM-YYYY, DD/MM/YY, etc.)
                    date_pattern = r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})'
                    # Regex for Amount (detect numbers with decimals)
                    amount_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}))'
                    
                    for line in lines:
                        if re.search(date_pattern, line) and re.search(amount_pattern, line):
                            # This looks like a transaction row - wrap it in a simple dict
                            transactions.append({"raw_text": line.strip()})
                    
                # STAGE 3: AI-Assisted Parsing (The "Universal" Fallback)
                # Lower threshold to trigger AI parsing for tricky pages
                if not transactions and text and len(text.strip()) > 20 and groq_api_key:
                    print(f"DEBUG: Triggering AI Layout Engine for Page {page_num+1}...")
                    ai_tx = _extract_with_ai(text, groq_api_key)
                    if ai_tx:
                        transactions.extend(ai_tx)
                        
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    
    return transactions

def _extract_with_ai(page_text, api_key):
    """Uses Groq to convert messy text into a structured transaction list."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
        
        prompt = f"""
        Extract a list of financial transactions from the following bank statement text.
        Return ONLY a JSON list of objects. Each object should have keys: "Date", "Particulars", "Amount".
        If 'Amount' is split into Debit/Credit, merge them into one 'Amount' key but mark negative for withdrawals.
        
        TEXT:
        {page_text}
        
        JSON OUTPUT:
        """
        response = llm.invoke(prompt)
        # Extract JSON block
        json_str = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
    except Exception as e:
        print(f"AI Extraction Error: {e}")
    return []

def process_statement_with_rag(pdf_path, groq_api_key):
    """
    Sets up a RAG pipeline to analyze bank transactions for fraud using Groq.
    """
    if not groq_api_key:
        return "Error: Groq API Key is missing. Please provide it in the sidebar."

    # Setup LLM (Groq) and Local Embeddings (HuggingFace)
    try:
        # Local embeddings (all-MiniLM-L6-v2 is small and fast)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Groq LLM (using Llama 3.3 70B for high performance)
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.1
        )
    except Exception as e:
        return f"Error initializing AI models: {e}"
    
    # Extract transactions using the new multi-stage engine
    tx_list = extract_transactions_from_pdf(pdf_path, groq_api_key)
    if not tx_list:
        return "No transactions found in the uploaded file. Please ensure it is a valid bank statement."
    
    # Prepare text for RAG
    texts = [desc for desc in [ " | ".join([f"{k}: {v}" for k, v in tx.items() if v]) for tx in tx_list] if desc.strip()]
    
    print(f"Extracted {len(tx_list)} transactions from {pdf_path}. Processing with Groq...")

    if not texts:
        return "Error: Could not extract any meaningful text from transactions for analysis."

    # Create Vector Store
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        return f"Error creating vector store: {e}"
    
    # Define Prompt
    prompt_template = """
    You are an expert financial fraud investigator. Analyze the following bank transactions extracted from a statement.
    Identify any transactions that look suspicious, fraudulent, or represent potential scams. 

    Look for:
    - Unusual transfer descriptions (lotteries, unexpected gifts, emergency requests).
    - Frequent small transfers to unknown digital wallets or people.
    - Large withdrawals or transfers that deviate from the normal pattern.
    - Inconsistent balance changes or multiple failed attempts (if visible).

    Context (Extracted Transactions):
    {context}
    
    Question: {question}
    
    Provide your analysis in a structured, professional format:
    ### 1. Overall Risk Assessment
    - [Safe / Low Risk / Medium Risk / High Risk]
    - Summary of the account health and risk status.

    ### 2. Suspicious Transactions Found
    - List each transaction with EXACT NAME/DESCRIPTION (Particulars), Date, Amount, and the specific reason for suspicion.
    - DO NOT skip the name of the transaction; it is critical for identification.
    - If no suspicious transactions are found, state "No suspicious transactions detected."

    ### 3. Detailed Reasoning
    - Explain WHY these transactions are flagged (e.g., "Frequent transfers to known high-risk keywords").

    ### 4. Recommendations
    - Specific steps for the user (e.g., "Contact your bank to dispute transaction on Date X", "Lock your card").
    """
    
    try:
        # Retrieve the most relevant transactions based on query
        # We increase k to see a larger set of transactions for comprehensive audit
        docs = vector_store.similarity_search("suspicious fraudulent activity transaction", k=min(len(texts), 60))
        context_text = "\n".join([doc.page_content for doc in docs])
        
        # Prepare the full prompt
        full_prompt = prompt_template.format(
            context=context_text, 
            question="Audit all transactions and highlight any signs of scams or fraud."
        )
        
        # Invoke Groq LLM
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"Error during Groq analysis: {e}"

if __name__ == "__main__":
    # Test script
    API_KEY = os.getenv("GROQ_API_KEY")
    if API_KEY:
        print("Running test analysis with Groq...")
        results = process_statement_with_rag("demo statement.pdf", API_KEY)
        print(results)
    else:
        print("Set GROQ_API_KEY in .env to run test.")

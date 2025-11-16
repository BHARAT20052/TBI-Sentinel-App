from dotenv import load_dotenv
import streamlit as st
import os
import json
from segment import segment_image
from forecast import forecast_vitals
from report import TBIReport 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 

# Load environment variables (OPENAI_API_KEY and OPENAI_API_BASE must be set for Agent Router)
load_dotenv()

# --- LLM and Output Setup ---

llm = ChatOpenAI(
    # Using the supported Claude Sonnet model from your Agent Router list.
    model="claude-sonnet-4-5-20250929", 
    api_key=os.getenv("OPENAI_API_KEY"),
    # CRITICAL FIX: Ensure the custom endpoint URL is explicitly set here
    openai_api_base=os.getenv("OPENAI_API_BASE") 
)

# Use StrOutputParser to reliably get the raw JSON string text.
parser = StrOutputParser() 

# Define the prompt to explicitly ask for JSON output and provide the Pydantic schema text
prompt_template = """
You are a specialized Military Medical AI assistant. Your task is to analyze the provided TBI data and generate a structured clinical report in JSON format that STRICTLY follows this Python Pydantic Schema structure:

{pydantic_schema}

Analyze the following TBI data and provide a detailed report as a single, valid JSON object:
Brain Anomaly: {anomaly}%
Risk Level: {risk}
Forecast: {forecast}
"""

prompt = ChatPromptTemplate.from_template(prompt_template).partial(
    # Pass the schema instructions as a string for the LLM to read
    pydantic_schema=TBIReport.schema_json(indent=2)
)

# The chain is defined as: Prompt | LLM | Parser (Returns JSON text string)
chain = prompt | llm | parser

# --- Web App ---

st.set_page_config(page_title="TBI Sentinel", layout="centered")
st.title("TBI Sentinel: Field TBI Analysis")

scan = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "png"])
vitals = st.file_uploader("Upload Vitals Data (CSV)", type="csv")

if scan and vitals:
    # 1. File Handling (Outside try block)
    temp_scan_path = "temp_scan.jpg"
    temp_vitals_path = "temp_vitals.csv"
    
    with open(temp_scan_path, "wb") as f: 
        f.write(scan.getbuffer())
    with open(temp_vitals_path, "wb") as f: 
        f.write(vitals.getbuffer())
    
    st.image(temp_scan_path, caption="Uploaded MRI Scan", use_column_width=True)
    
    # 2. Run AI Analysis (Outside try block)
    st.info("Running image segmentation and vital forecasting...")
    anomaly = segment_image(temp_scan_path)
    forecast_data = forecast_vitals(temp_vitals_path, anomaly['volume_percent']) 
    
    # 3. Generate Report using LLM Chain (Isolated try block)
    st.info("Generating Structured Clinical Report...")
    
    final_report_json = None 
    
    try:
        final_report_json = chain.invoke({
            "anomaly": anomaly['volume_percent'],
            "risk": forecast_data['risk'],
            "forecast": forecast_data['forecast']
        })
        
    except Exception as e:
        # If the LLM call fails, report it and set a placeholder text
        st.error(f"An error occurred during report generation: {e}")
        st.warning(f"Error Code 401: Please contact Agent Router support (via the link in the error message) as the issue is with the key's permissions, not its expiration.")
        final_report_json = "{}"

    # 4. Display Results
    if final_report_json is not None:
        st.success("Analysis Complete!")
        st.write("### AI Structured Clinical Report")   
        st.json(final_report_json) 
    # 5. Display Forecast Image
    if os.path.exists("forecast.png"):
        st.image("forecast.png", caption="48-Hour Heart Rate Forecast")
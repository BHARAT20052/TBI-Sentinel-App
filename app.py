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
    api_key=os.getenv("OPENAI_API_KEY")    
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
    
    # Initialize the output variable with a new name and value outside the try block
    final_report_json = None 
    
    try:
        # LLM call is the only piece that is truly network-dependent and needs protection
        final_report_json = chain.invoke({
            "anomaly": anomaly['volume_percent'],
            "risk": forecast_data['risk'],
            "forecast": forecast_data['forecast']
        })
        
    except Exception as e:
        # If the LLM call fails, report it and set a placeholder text
        st.error(f"An error occurred during report generation: {e}")
        st.warning("Please verify your Streamlit secrets (OPENAI_API_KEY and OPENAI_API_BASE) are set correctly for Agent Router.")
        # Crucial: Set output to an empty JSON string on failure
        final_report_json = "{}"

    # 4. Display Results
    if final_report_json is not None:
        st.success("Analysis Complete!")
        st.write("### AI Structured Clinical Report")
        
        # Display the string directly. This bypasses the error.
        st.json(final_report_json) 

    # 5. Display Forecast Image
    if os.path.exists("forecast.png"):
        st.image("forecast.png", caption="48-Hour Heart Rate Forecast")

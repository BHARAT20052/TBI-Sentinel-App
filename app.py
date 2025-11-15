from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from segment import segment_image
from forecast import forecast_vitals
from report import TBIReport 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser # <-- DELETED: Not needed for JSON structure
# from langchain_core.output_parsers import JsonOutputParser # <-- DELETED: Using LLM built-in structure
import os

# --- LLM and Structured Output Setup ---

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct", 
    api_key=os.getenv("OPENAI_API_KEY")    
)

# CRITICAL FIX: Use with_structured_output directly on the LLM
# This is the modern, robust way to get Pydantic JSON from the LLM
structured_llm = llm.with_structured_output(TBIReport)

prompt = ChatPromptTemplate.from_messages( 
    [
        # Removed format_instructions from the system message as the LLM handles it now
        ("system", "You are a specialized Military Medical AI assistant. Your task is to analyze the provided TBI data and generate a structured clinical report in JSON format, strictly following the provided schema."),
        ("human", "Analyze the following TBI data and provide a detailed report:\nBrain Anomaly: {anomaly}%\nRisk Level: {risk}\nForecast: {forecast}"),
    ]
) # Removed the .partial(format_instructions...) call

# The chain now produces a structured TBIReport object directly
chain = prompt | structured_llm

# --- Web App ---

# Assuming imports for segment, forecast, and report.py are correct
# ... (rest of the app.py file)

st.set_page_config(page_title="TBI Sentinel", layout="centered")
st.title("TBI Sentinel: Field TBI Analysis")

# --- FILE UPLOAD AND PROCESSING ---
# Assuming temporary file handling exists here...

scan = st.file_uploader("Upload Brain MRI", type=["jpg", "png"])
vitals = st.file_uploader("Upload Vitals CSV", type="csv")
if scan and vitals:
    # Save files to temp for processing (assuming the helper functions handle this)
    with open("temp_scan.jpg", "wb") as f: f.write(scan.getbuffer())
    with open("temp_vitals.csv", "wb") as f: f.write(vitals.getbuffer())
    
    st.image("temp_scan.jpg", caption="Uploaded MRI Scan", width=300)
    
    # Run AI
    anomaly = segment_image("temp_scan.jpg")
    # Note: forecast_vitals must now return 'risk_score' as defined in the next step
    forecast_data = forecast_vitals("temp_vitals.csv", anomaly['volume_percent']) 
    
    # Generate report
    st.info("Generating Structured Clinical Report...")
    try:
        report: TBIReport = chain.invoke({
            "anomaly": anomaly['volume_percent'],
            "risk": forecast_data['risk'],
            "forecast": forecast_data['forecast']
        })
        
        st.success("Analysis Complete!")
        st.write("### AI Structured Clinical Report")
        
        # Display structured data nicely using the correct Pydantic method
        # This will now definitely be a TBIReport object thanks to with_structured_output
        st.json(report.model_dump())
        
        st.image("forecast.png", caption="48-Hour Heart Rate Forecast")
        
    except Exception as e:
        st.error(f"Error generating report: {e}")
        st.error("This usually means the LLM failed to produce valid JSON or the Agent Router failed to connect properly.")
        st.info("Please check your Streamlit secrets again (API Key and OPENAI_API_BASE).")

# --- END OF WEB APP ---
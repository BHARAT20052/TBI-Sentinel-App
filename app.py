# app.py (Modified)
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from segment import segment_image
from forecast import forecast_vitals
from report import TBIReport # <-- NEW: Import the Pydantic model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser # <-- CHANGE: Use JSON parser
import os

llm = ChatOpenAI(
    # FINAL FIX: Change model name to OpenRouter's preferred format
    model="openai/gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1" 
)
parser = JsonOutputParser(pydantic_object=TBIReport) # Initialize Pydantic parser

prompt = ChatPromptTemplate.from_messages( # Use a system role for robustness
    [
        ("system", "You are a specialized Military Medical AI assistant. Your task is to analyze the provided TBI data and generate a structured clinical report in JSON format, strictly following the provided schema. {format_instructions}"),
        ("human", "Analyze the following TBI data and provide a detailed report:\nBrain Anomaly: {anomaly}%\nRisk Level: {risk}\nForecast: {forecast}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) # Pass format instructions

chain = prompt | llm | parser # Chain now produces a structured Python object

# --- Web App ---
# ... (rest of the app.py file)
# ...
scan = st.file_uploader("Upload Brain MRI", type=["jpg", "png"])
vitals = st.file_uploader("Upload Vitals CSV", type="csv")
if scan and vitals:
    # ... (file saving) ...
    
    # Run AI
    anomaly = segment_image("temp_scan.jpg")
    # Note: forecast_vitals must now return 'risk_score' as defined in the next step
    forecast_data = forecast_vitals("temp_vitals.csv", anomaly['volume_percent']) 
    
    # Generate report
    st.info("Generating Structured Clinical Report...")
    report = chain.invoke({
        "anomaly": anomaly['volume_percent'],
        "risk": forecast_data['risk'],
        "forecast": forecast_data['forecast']
        })
    
    st.success("Analysis Complete!")
    st.write("### AI Structured Clinical Report")
    
    # Display structured data nicely
    st.json(report.dict() if hasattr(report, 'dict') else report)
    
    st.image("forecast.png", caption="48-Hour Heart Rate Forecast")
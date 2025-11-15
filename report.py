# report_schema.py content
from pydantic import BaseModel, Field

class TBIReport(BaseModel):
    """A structured clinical recommendation report for a soldier with TBI."""

    risk_level: str = Field(description="Overall risk (CRITICAL, HIGH, MODERATE, LOW).")
    risk_justification: str = Field(description="A brief, 1-2 sentence justification for the risk level, mentioning volume and heart rate trend.")

    anomaly_volume_percent: float = Field(description="The calculated percentage of TBI anomaly volume.")
    vitals_trend: str = Field(description="The forecast trend of the heart rate (e.g., Stable, Rising, High Variability).")

    field_recommendation: str = Field(description="An urgent, practical field recommendation (e.g., 'Immobilize C-spine and request immediate MEDEVAC').")

    monitoring_note: str = Field(description="A brief note on what vitals should be monitored next 4 hours.")
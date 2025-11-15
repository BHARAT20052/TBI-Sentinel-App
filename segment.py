from transformers import pipeline
import numpy as np

def segment_image(path):
    segmenter = pipeline("image-segmentation", model="openmmlab/upernet-convnext-small")
    result = segmenter(path)
    volume = np.random.uniform(5, 25)  # Simulated TBI volume
    return {"anomaly": result, "volume_percent": round(volume, 2)}
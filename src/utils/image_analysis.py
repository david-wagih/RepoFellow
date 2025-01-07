from typing import Dict, Any
import base64
from PIL import Image
import io
from langchain_openai import ChatOpenAI


def analyze_design_image(image_data: bytes) -> str:
    """Analyze design image using vision model"""
    try:
        # Convert image data to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Initialize vision model
        vision_model = ChatOpenAI(model="gpt-4-vision-preview")

        # Analyze image
        response = vision_model.invoke(
            [
                {
                    "type": "text",
                    "text": "Analyze this design screenshot and describe the UI components, layout, and functionality in detail.",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                },
            ]
        )

        return response.content
    except Exception as e:
        raise Exception(f"Image analysis failed: {str(e)}")

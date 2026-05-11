from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)

model = None

try:
    model = load_model(
        "models/image_model.keras",
        compile=False
    )
    logger.info("✅ Model loaded")

except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None

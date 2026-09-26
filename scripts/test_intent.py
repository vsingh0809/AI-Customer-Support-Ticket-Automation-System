from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.rag.generator import DeepSeekProvider
from app.core.config import get_settings

settings = get_settings()

provider = DeepSeekProvider(
    api_key=settings.deepseek_api_key or "",
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    max_tokens=settings.llm_max_tokens,
)

classifier = IntentClassifier(provider)

result = classifier.classify(
    "Where is my order 45821?"
)

print(result.model_dump())
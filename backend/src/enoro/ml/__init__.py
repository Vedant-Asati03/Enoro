"""
ML package for Enoro content analysis and recommendations.

Restructured for scalability:
- content_analysis/ - Text processing, topic modeling, tag generation
- recommendations/ - Recommendation systems (collaborative, content-based, hybrid)
- shared/ - Common utilities (data preprocessing, model management, evaluation)
- models/ - Trained model storage
"""

# Content Analysis Components
from backend.src.enoro.ml.content_analysis import (
    FeatureExtractor,
    TextFeatures,
    ContentAnalysis,
    TopicModeler,
    TopicInfo,
    UserProfile,
    TagGenerator,
    TagSuggestion,
    ChannelTagging,
)

# Shared Utilities
from backend.src.enoro.ml.shared import (
    DataPreprocessor,
    ModelManager,
    MetricsCalculator,
)

# Recommendation Systems
from backend.src.enoro.ml.recommendations import (
    CollaborativeFilter,
    ContentBasedRecommender,
    HybridRecommender,
    RecommendationEngine,
)

__all__ = [
    # Content Analysis
    "FeatureExtractor",
    "TextFeatures",
    "ContentAnalysis",
    "TopicModeler",
    "TopicInfo",
    "UserProfile",
    "TagGenerator",
    "TagSuggestion",
    "ChannelTagging",
    # Shared Utilities
    "DataPreprocessor",
    "ModelManager",
    "MetricsCalculator",
    # Recommendation Systems
    "CollaborativeFilter",
    "ContentBasedRecommender",
    "HybridRecommender",
    "RecommendationEngine",
]

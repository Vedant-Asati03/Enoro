"""
Recommendation Systems package for personalized content discovery.

This package contains various recommendation algorithms:
- Collaborative Filtering (user-based, item-based, matrix factorization)
- Content-Based Filtering (tags, features, preferences)
- Hybrid Recommendation Systems (combining multiple approaches)
- Main Recommendation Engine (unified interface)
"""

from backend.src.enoro.ml.recommendations.collaborative_filtering import (
    CollaborativeFilter,
)
from backend.src.enoro.ml.recommendations.content_based import ContentBasedRecommender
from backend.src.enoro.ml.recommendations.hybrid import HybridRecommender
from backend.src.enoro.ml.recommendations.recommendation_engine import (
    RecommendationEngine,
)

__all__ = [
    "CollaborativeFilter",
    "ContentBasedRecommender",
    "HybridRecommender",
    "RecommendationEngine",
]

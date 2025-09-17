"""
Main Recommendation Engine.

Orchestrates all recommendation systems and provides a unified interface
for generating personalized channel recommendations.
"""

from typing import List, Dict, Tuple, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from backend.src.enoro.ml.recommendations.hybrid import HybridRecommender
from backend.src.enoro.database.models.channel import UserSubscription
from backend.src.enoro.database.models.search import UserProfile


class RecommendationEngine:
    """
    Main recommendation engine that coordinates all recommendation systems.

    Features:
    1. Unified interface for all recommendation types
    2. Context-aware recommendations
    3. Recommendation caching and optimization
    4. A/B testing support for different algorithms
    5. Performance monitoring and analytics
    """

    def __init__(self):
        """Initialize the recommendation engine."""
        self.hybrid_recommender = HybridRecommender()
        self.last_training_time = None
        self.recommendation_cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache recommendations for 1 hour

    def initialize(self, db: Session) -> bool:
        """
        Initialize and train all recommendation models.

        Args:
            db: Database session

        Returns:
            True if initialization was successful
        """
        try:
            print("Initializing recommendation engine...")

            success = self.hybrid_recommender.fit(db)

            if success:
                self.last_training_time = datetime.now()
                print("Recommendation engine initialized successfully")
                return True
            else:
                print("Warning: Recommendation engine initialization had issues")
                return False

        except Exception as e:
            print(f"Error initializing recommendation engine: {e}")
            return False

    def get_recommendations(
        self,
        db: Session,
        user_id: str,
        n_recommendations: int = 10,
        strategy: str = "auto",
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user.

        Args:
            db: Database session
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            strategy: Recommendation strategy ('auto', 'hybrid', 'collaborative', 'content', 'popularity')
            use_cache: Whether to use cached recommendations

        Returns:
            List of recommendation dictionaries
        """
        # Check cache first
        if use_cache:
            cached_recs = self._get_cached_recommendations(
                user_id, n_recommendations, strategy
            )
            if cached_recs:
                return cached_recs

        try:
            # Determine strategy if auto
            if strategy == "auto":
                strategy = self._determine_best_strategy(db, user_id)

            # Get recommendations based on strategy
            if strategy == "hybrid":
                recommendations = (
                    self.hybrid_recommender.get_recommendations_with_explanations(
                        db, user_id, n_recommendations
                    )
                )
            else:
                # Get basic recommendations and convert to detailed format
                basic_recs = self.hybrid_recommender.get_recommendations(
                    db, user_id, n_recommendations, strategy
                )
                recommendations = self._convert_to_detailed_format(db, basic_recs)

            # Add context and metadata
            recommendations = self._add_recommendation_metadata(
                db, user_id, recommendations
            )

            # Cache results
            if use_cache:
                self._cache_recommendations(
                    user_id, n_recommendations, strategy, recommendations
                )

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def _determine_best_strategy(self, db: Session, user_id: str) -> str:
        """
        Determine the best recommendation strategy for a user based on their data.

        Args:
            db: Database session
            user_id: Target user ID

        Returns:
            Best strategy name
        """
        try:
            # Get user subscription count
            subscription_count = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .count()
            )

            # Get user profile if available
            user_profile = (
                db.query(UserProfile).filter(UserProfile.id == user_id).first()
            )

            # Decision logic based on available data
            if subscription_count >= 5:
                # Enough subscription data for hybrid approach
                return "hybrid"
            elif subscription_count >= 2:
                # Some subscription data, prefer content-based
                return "content"
            elif user_profile and getattr(user_profile, "preferred_categories", None):
                # User has preferences but few subscriptions
                return "content"
            else:
                # New user, use popularity
                return "popularity"

        except Exception as e:
            print(f"Error determining strategy: {e}")
            return "hybrid"

    def _convert_to_detailed_format(
        self, db: Session, basic_recommendations: List[Tuple[str, float, str]]
    ) -> List[Dict[str, Any]]:
        """Convert basic recommendations to detailed format."""
        from backend.src.enoro.database.models.channel import Channel

        detailed_recommendations = []

        for channel_id, score, source in basic_recommendations:
            try:
                # Get channel information
                channel = db.query(Channel).filter(Channel.id == channel_id).first()

                if channel:
                    detailed_recommendations.append(
                        {
                            "channel_id": channel_id,
                            "channel_name": str(getattr(channel, "name", "Unknown")),
                            "score": float(score),
                            "source": source,
                            "explanation": f"Recommended via {source} algorithm",
                            "subscriber_count": getattr(channel, "subscriber_count", 0)
                            or 0,
                            "video_count": getattr(channel, "video_count", 0) or 0,
                            "description": str(
                                getattr(channel, "description", "") or ""
                            )[:200]
                            + "..."
                            if getattr(channel, "description", "")
                            else "",
                        }
                    )
            except Exception as e:
                print(f"Error converting recommendation for channel {channel_id}: {e}")
                continue

        return detailed_recommendations

    def _add_recommendation_metadata(
        self, db: Session, user_id: str, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add additional metadata to recommendations."""
        try:
            for rec in recommendations:
                # Add recommendation metadata
                rec["recommended_at"] = datetime.now().isoformat()
                rec["user_id"] = user_id
                rec["engine_version"] = "1.0"

                # Add confidence score based on source
                if "hybrid" in rec.get("source", ""):
                    rec["confidence"] = min(rec["score"] * 1.2, 1.0)
                elif "collaborative" in rec.get("source", ""):
                    rec["confidence"] = min(rec["score"] * 1.1, 1.0)
                else:
                    rec["confidence"] = rec["score"]

            return recommendations

        except Exception as e:
            print(f"Error adding metadata: {e}")
            return recommendations

    def _get_cached_recommendations(
        self, user_id: str, n_recommendations: int, strategy: str
    ) -> List[Dict[str, Any]]:
        """Get cached recommendations if available and not expired."""
        cache_key = f"{user_id}_{n_recommendations}_{strategy}"

        if cache_key in self.recommendation_cache:
            cached_entry = self.recommendation_cache[cache_key]
            cache_time = cached_entry["timestamp"]

            if datetime.now() - cache_time < self.cache_duration:
                return cached_entry["recommendations"]

        return []

    def _cache_recommendations(
        self,
        user_id: str,
        n_recommendations: int,
        strategy: str,
        recommendations: List[Dict[str, Any]],
    ):
        """Cache recommendations for future use."""
        cache_key = f"{user_id}_{n_recommendations}_{strategy}"

        self.recommendation_cache[cache_key] = {
            "timestamp": datetime.now(),
            "recommendations": recommendations,
        }

        # Clean old cache entries to prevent memory bloat
        self._clean_cache()

    def _clean_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []

        for key, entry in self.recommendation_cache.items():
            if current_time - entry["timestamp"] > self.cache_duration:
                expired_keys.append(key)

        for key in expired_keys:
            del self.recommendation_cache[key]

    def get_similar_channels(
        self, db: Session, channel_id: str, n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get channels similar to a specific channel.

        Args:
            db: Database session
            channel_id: Target channel ID
            n_recommendations: Number of similar channels to return

        Returns:
            List of similar channel dictionaries
        """
        try:
            if not self.hybrid_recommender.content_recommender.is_trained():
                return []

            # Get similar channels from content-based recommender
            similar_channels = (
                self.hybrid_recommender.content_recommender.get_similar_channels(
                    channel_id, n_recommendations
                )
            )

            # Convert to detailed format
            return self._convert_similar_channels_to_detailed_format(
                db, similar_channels
            )

        except Exception as e:
            print(f"Error getting similar channels: {e}")
            return []

    def _convert_similar_channels_to_detailed_format(
        self, db: Session, similar_channels: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Convert similar channels to detailed format."""
        from backend.src.enoro.database.models.channel import Channel

        detailed_channels = []

        for channel_id, similarity_score in similar_channels:
            try:
                channel = db.query(Channel).filter(Channel.id == channel_id).first()

                if channel:
                    detailed_channels.append(
                        {
                            "channel_id": channel_id,
                            "channel_name": str(getattr(channel, "name", "Unknown")),
                            "similarity_score": float(similarity_score),
                            "subscriber_count": getattr(channel, "subscriber_count", 0)
                            or 0,
                            "video_count": getattr(channel, "video_count", 0) or 0,
                            "description": str(
                                getattr(channel, "description", "") or ""
                            )[:200]
                            + "..."
                            if getattr(channel, "description", "")
                            else "",
                        }
                    )
            except Exception as e:
                print(f"Error converting similar channel {channel_id}: {e}")
                continue

        return detailed_channels

    def get_recommendation_stats(self, db: Session, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive recommendation statistics for a user.

        Args:
            db: Database session
            user_id: Target user ID

        Returns:
            Dictionary with recommendation statistics
        """
        try:
            stats = {
                "engine_status": {
                    "is_trained": self.hybrid_recommender.is_trained(),
                    "last_training": self.last_training_time.isoformat()
                    if self.last_training_time
                    else None,
                    "cache_size": len(self.recommendation_cache),
                }
            }

            # Add hybrid recommender stats
            hybrid_stats = self.hybrid_recommender.get_recommendation_stats(db, user_id)
            stats.update(hybrid_stats)

            # Add user subscription info
            subscription_count = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .count()
            )

            stats["user_data"] = {
                "subscription_count": subscription_count,
                "recommended_strategy": self._determine_best_strategy(db, user_id),
            }

            return stats

        except Exception as e:
            print(f"Error getting recommendation stats: {e}")
            return {}

    def retrain_models(self, db: Session) -> bool:
        """
        Retrain all recommendation models with latest data.

        Args:
            db: Database session

        Returns:
            True if retraining was successful
        """
        try:
            print("Retraining recommendation models...")

            # Clear cache since models will change
            self.recommendation_cache.clear()

            # Retrain hybrid recommender
            success = self.hybrid_recommender.fit(db)

            if success:
                self.last_training_time = datetime.now()
                print("Models retrained successfully")
            else:
                print("Warning: Model retraining had issues")

            return success

        except Exception as e:
            print(f"Error retraining models: {e}")
            return False

    def clear_cache(self):
        """Clear all cached recommendations."""
        self.recommendation_cache.clear()

    def is_ready(self) -> bool:
        """Check if the recommendation engine is ready to serve recommendations."""
        return self.hybrid_recommender.is_trained()

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the recommendation engine."""
        return {
            "status": "healthy" if self.is_ready() else "not_ready",
            "models_trained": {
                "collaborative": self.hybrid_recommender.collaborative_filter.is_trained(),
                "content": self.hybrid_recommender.content_recommender.is_trained(),
                "popularity": len(self.hybrid_recommender.popular_channels) > 0,
            },
            "last_training": self.last_training_time.isoformat()
            if self.last_training_time
            else None,
            "cache_size": len(self.recommendation_cache),
            "ready_for_recommendations": self.is_ready(),
        }

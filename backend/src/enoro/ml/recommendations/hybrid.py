"""
Hybrid Recommendation System.

Combines multiple recommendation approaches to provide better suggestions.
Integrates collaborative filtering, content-based, and other recommendation methods.
"""

from typing import List, Dict, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.src.enoro.ml.recommendations.collaborative_filtering import (
    CollaborativeFilter,
)
from backend.src.enoro.ml.recommendations.content_based import ContentBasedRecommender


class HybridRecommender:
    """
    Hybrid recommendation system combining multiple approaches.

    Features:
    1. Collaborative Filtering (user-based, item-based, matrix factorization)
    2. Content-Based Filtering (tags, features, preferences)
    3. Popularity-Based Recommendations (fallback)
    4. Weighted combination strategies
    5. Context-aware recommendations
    """

    def __init__(
        self,
        collaborative_weight: float = 0.5,
        content_weight: float = 0.4,
        popularity_weight: float = 0.1,
    ):
        """
        Initialize hybrid recommender.

        Args:
            collaborative_weight: Weight for collaborative filtering recommendations
            content_weight: Weight for content-based recommendations
            popularity_weight: Weight for popularity-based recommendations
        """
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight

        # Normalize weights
        total_weight = collaborative_weight + content_weight + popularity_weight
        if total_weight > 0:
            self.collaborative_weight /= total_weight
            self.content_weight /= total_weight
            self.popularity_weight /= total_weight

        # Initialize component recommenders
        self.collaborative_filter = CollaborativeFilter()
        self.content_recommender = ContentBasedRecommender()

        # Cache for popular channels
        self.popular_channels: List[Tuple[str, float]] = []

    def fit(self, db: Session) -> bool:
        """
        Train all recommendation models.

        Args:
            db: Database session

        Returns:
            True if training was successful
        """
        success_count = 0

        try:
            # Train collaborative filtering
            if self.collaborative_filter.fit(db):
                print("Collaborative filtering trained successfully")
                success_count += 1
            else:
                print("Warning: Collaborative filtering training failed")
        except Exception as e:
            print(f"Error training collaborative filtering: {e}")

        try:
            # Train content-based recommender
            if self.content_recommender.fit(db):
                print("Content-based recommender trained successfully")
                success_count += 1
            else:
                print("Warning: Content-based recommender training failed")
        except Exception as e:
            print(f"Error training content-based recommender: {e}")

        try:
            # Calculate popular channels
            self._calculate_popular_channels(db)
            print("Popular channels calculated successfully")
            success_count += 1
        except Exception as e:
            print(f"Error calculating popular channels: {e}")

        return success_count > 0

    def _calculate_popular_channels(self, db: Session):
        """Calculate popularity scores for channels based on subscriber count and engagement."""
        from backend.src.enoro.database.models.channel import Channel, UserSubscription

        # Get channel popularity metrics
        channel_stats = {}

        # Count subscriptions per channel
        subscription_counts = (
            db.query(
                UserSubscription.channel_id,
                func.count(UserSubscription.id).label("subscription_count"),
            )
            .group_by(UserSubscription.channel_id)
            .all()
        )

        for channel_id, count in subscription_counts:
            channel_stats[str(channel_id)] = {"subscriptions": count}

        # Get channel subscriber counts
        channels = db.query(Channel).all()
        for channel in channels:
            channel_id = str(channel.id)
            subscriber_count = getattr(channel, "subscriber_count", 0) or 0

            if channel_id not in channel_stats:
                channel_stats[channel_id] = {"subscriptions": 0}

            channel_stats[channel_id]["subscriber_count"] = subscriber_count

        # Calculate popularity scores
        popular_channels = []
        for channel_id, stats in channel_stats.items():
            subscriptions = stats.get("subscriptions", 0)
            subscriber_count = stats.get("subscriber_count", 0)

            # Combine subscription count and subscriber count for popularity score
            # Normalize to prevent extremely large numbers from dominating
            normalized_subscribers = min(subscriber_count / 1000000, 1.0)  # Cap at 1M
            normalized_subscriptions = min(subscriptions / 100, 1.0)  # Cap at 100

            popularity_score = (
                0.3 * normalized_subscribers + 0.7 * normalized_subscriptions
            )

            if popularity_score > 0:
                popular_channels.append((channel_id, popularity_score))

        # Sort by popularity and cache
        self.popular_channels = sorted(
            popular_channels, key=lambda x: x[1], reverse=True
        )

    def get_recommendations(
        self,
        db: Session,
        user_id: str,
        n_recommendations: int = 10,
        strategy: str = "hybrid",
    ) -> List[Tuple[str, float, str]]:
        """
        Get recommendations using specified strategy.

        Args:
            db: Database session
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            strategy: Recommendation strategy ('hybrid', 'collaborative', 'content', 'popularity')

        Returns:
            List of (channel_id, score, source) tuples
        """
        if strategy == "collaborative":
            return self._get_collaborative_recommendations(user_id, n_recommendations)
        elif strategy == "content":
            return self._get_content_recommendations(db, user_id, n_recommendations)
        elif strategy == "popularity":
            return self._get_popularity_recommendations(db, user_id, n_recommendations)
        else:  # hybrid
            return self._get_hybrid_recommendations(db, user_id, n_recommendations)

    def _get_collaborative_recommendations(
        self, user_id: str, n_recommendations: int
    ) -> List[Tuple[str, float, str]]:
        """Get recommendations from collaborative filtering."""
        if not self.collaborative_filter.is_trained():
            return []

        try:
            # Get combined collaborative filtering recommendations
            recommendations = self.collaborative_filter.get_combined_recommendations(
                user_id, n_recommendations
            )

            return [
                (channel_id, score, "collaborative")
                for channel_id, score in recommendations
            ]

        except Exception as e:
            print(f"Error getting collaborative recommendations: {e}")
            return []

    def _get_content_recommendations(
        self, db: Session, user_id: str, n_recommendations: int
    ) -> List[Tuple[str, float, str]]:
        """Get recommendations from content-based filtering."""
        if not self.content_recommender.is_trained():
            return []

        try:
            # Get content-based recommendations
            recommendations = self.content_recommender.get_user_recommendations(
                db, user_id, n_recommendations
            )

            return [
                (channel_id, score, "content") for channel_id, score in recommendations
            ]

        except Exception as e:
            print(f"Error getting content recommendations: {e}")
            return []

    def _get_popularity_recommendations(
        self, db: Session, user_id: str, n_recommendations: int
    ) -> List[Tuple[str, float, str]]:
        """Get recommendations based on popularity."""
        try:
            from backend.src.enoro.database.models.channel import UserSubscription

            # Get user's subscriptions to exclude
            subscriptions = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .all()
            )

            subscribed_channels = {str(sub.channel_id) for sub in subscriptions}

            # Filter out already subscribed channels
            filtered_popular = [
                (channel_id, score, "popularity")
                for channel_id, score in self.popular_channels
                if channel_id not in subscribed_channels
            ]

            return filtered_popular[:n_recommendations]

        except Exception as e:
            print(f"Error getting popularity recommendations: {e}")
            return []

    def _get_hybrid_recommendations(
        self, db: Session, user_id: str, n_recommendations: int
    ) -> List[Tuple[str, float, str]]:
        """Get hybrid recommendations combining all methods."""
        try:
            # Get recommendations from each method
            collaborative_recs = self._get_collaborative_recommendations(
                user_id, n_recommendations * 2
            )
            content_recs = self._get_content_recommendations(
                db, user_id, n_recommendations * 2
            )
            popularity_recs = self._get_popularity_recommendations(
                db, user_id, n_recommendations * 2
            )

            # Combine scores using weighted approach
            combined_scores = {}
            sources = {}

            # Add collaborative filtering scores
            for channel_id, score, source in collaborative_recs:
                combined_scores[channel_id] = (
                    combined_scores.get(channel_id, 0)
                    + self.collaborative_weight * score
                )
                if channel_id not in sources:
                    sources[channel_id] = [source]
                else:
                    sources[channel_id].append(source)

            # Add content-based scores
            for channel_id, score, source in content_recs:
                combined_scores[channel_id] = (
                    combined_scores.get(channel_id, 0) + self.content_weight * score
                )
                if channel_id not in sources:
                    sources[channel_id] = [source]
                else:
                    sources[channel_id].append(source)

            # Add popularity scores
            for channel_id, score, source in popularity_recs:
                combined_scores[channel_id] = (
                    combined_scores.get(channel_id, 0) + self.popularity_weight * score
                )
                if channel_id not in sources:
                    sources[channel_id] = [source]
                else:
                    sources[channel_id].append(source)

            # Create combined source labels
            combined_sources = {}
            for channel_id, source_list in sources.items():
                unique_sources = list(set(source_list))
                combined_sources[channel_id] = "+".join(sorted(unique_sources))

            # Sort by combined score
            sorted_recommendations = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Return with source information
            return [
                (channel_id, score, combined_sources.get(channel_id, "hybrid"))
                for channel_id, score in sorted_recommendations[:n_recommendations]
            ]

        except Exception as e:
            print(f"Error getting hybrid recommendations: {e}")
            return []

    def get_recommendations_with_explanations(
        self, db: Session, user_id: str, n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations with detailed explanations.

        Args:
            db: Database session
            user_id: Target user ID
            n_recommendations: Number of recommendations to return

        Returns:
            List of recommendation dictionaries with explanations
        """
        try:
            recommendations = self._get_hybrid_recommendations(
                db, user_id, n_recommendations
            )

            from backend.src.enoro.database.models.channel import Channel

            detailed_recommendations = []

            for channel_id, score, source in recommendations:
                # Get channel information
                channel = db.query(Channel).filter(Channel.id == channel_id).first()

                if channel:
                    # Generate explanation based on recommendation source
                    explanation = self._generate_explanation(
                        db, user_id, channel_id, source
                    )

                    detailed_recommendations.append(
                        {
                            "channel_id": channel_id,
                            "channel_name": str(getattr(channel, "name", "Unknown")),
                            "score": float(score),
                            "source": source,
                            "explanation": explanation,
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

            return detailed_recommendations

        except Exception as e:
            print(f"Error getting recommendations with explanations: {e}")
            return []

    def _generate_explanation(
        self, db: Session, user_id: str, channel_id: str, source: str
    ) -> str:
        """Generate explanation for why a channel was recommended."""
        try:
            if "collaborative" in source:
                explanation = "Recommended because users with similar preferences also subscribe to this channel."
            elif "content" in source:
                explanation = "Recommended based on similarity to channels you're already subscribed to."
            elif "popularity" in source:
                explanation = "Recommended because this is a popular channel in your areas of interest."
            else:
                explanation = "Recommended based on multiple factors including user preferences and content similarity."

            # Add specific details if content recommender is available
            if self.content_recommender.is_trained():
                channel_summary = self.content_recommender.get_channel_features_summary(
                    channel_id
                )
                if channel_summary and channel_summary.get("tags"):
                    top_tags = channel_summary["tags"][:3]
                    if top_tags:
                        explanation += (
                            f" This channel covers topics like: {', '.join(top_tags)}."
                        )

            return explanation

        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "Recommended based on your preferences."

    def get_recommendation_stats(self, db: Session, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about recommendation quality and coverage.

        Args:
            db: Database session
            user_id: Target user ID

        Returns:
            Dictionary with recommendation statistics
        """
        try:
            stats = {
                "user_id": user_id,
                "models_trained": {
                    "collaborative": self.collaborative_filter.is_trained(),
                    "content": self.content_recommender.is_trained(),
                    "popularity": len(self.popular_channels) > 0,
                },
                "weights": {
                    "collaborative": self.collaborative_weight,
                    "content": self.content_weight,
                    "popularity": self.popularity_weight,
                },
            }

            # Add user-specific stats from collaborative filter
            if self.collaborative_filter.is_trained():
                user_stats = self.collaborative_filter.get_user_statistics(user_id)
                stats["user_profile"] = user_stats

            # Add total number of popular channels
            stats["popular_channels_count"] = len(self.popular_channels)

            return stats

        except Exception as e:
            print(f"Error getting recommendation stats: {e}")
            return {}

    def update_weights(
        self,
        collaborative_weight: float,
        content_weight: float,
        popularity_weight: float,
    ):
        """Update recommendation weights and normalize them."""
        total_weight = collaborative_weight + content_weight + popularity_weight

        if total_weight > 0:
            self.collaborative_weight = collaborative_weight / total_weight
            self.content_weight = content_weight / total_weight
            self.popularity_weight = popularity_weight / total_weight

    def is_trained(self) -> bool:
        """Check if at least one recommendation model is trained."""
        return (
            self.collaborative_filter.is_trained()
            or self.content_recommender.is_trained()
            or len(self.popular_channels) > 0
        )

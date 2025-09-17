"""
Content-Based Recommendation System.

Generates channel recommendations based on content similarity,
using tags, features, and channel characteristics.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.tags import ChannelTag, ContentTag
from backend.src.enoro.database.models.search import UserPreferenceTags
from backend.src.enoro.ml.shared.data_preprocessing import DataPreprocessor


class ContentBasedRecommender:
    """
    Content-based recommendation system using channel features and tags.

    Recommends channels based on:
    1. Content tags and categories
    2. Channel features (subscriber count, description, etc.)
    3. User preference patterns
    4. Topic similarities
    """

    def __init__(self, max_features: int = 1000, min_df: int = 2, max_df: float = 0.8):
        """
        Initialize content-based recommender.

        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.preprocessor = DataPreprocessor()
        self.tfidf_vectorizer = None
        self.feature_scaler = None

        # Cached data
        self.channel_features: Dict[str, Dict] = {}
        self.channel_tags: Dict[str, List[str]] = {}
        self.tag_vectors = None
        self.feature_vectors = None
        self.channel_similarity_matrix = None
        self.channels_list: List[str] = []

    def fit(self, db: Session) -> bool:
        """
        Train the content-based recommendation models.

        Args:
            db: Database session

        Returns:
            True if training was successful
        """
        try:
            # Get all channels with their tags and features
            channels = db.query(Channel).all()

            if len(channels) < 2:
                print("Insufficient channels for content-based recommendations")
                return False

            self.channels_list = [str(channel.id) for channel in channels]

            # Extract channel features
            self.channel_features = self._extract_channel_features(db, channels)

            # Extract channel tags
            self.channel_tags = self._extract_channel_tags(db, channels)

            # Create TF-IDF vectors from tags and descriptions
            self._create_tag_vectors(channels)

            # Create feature vectors
            self._create_feature_vectors()

            # Calculate channel similarity matrix
            self._calculate_channel_similarity()

            print(f"Content-based recommender trained on {len(channels)} channels")
            return True

        except Exception as e:
            print(f"Error training content-based recommender: {e}")
            return False

    def _extract_channel_features(
        self, db: Session, channels: List[Channel]
    ) -> Dict[str, Dict]:
        """Extract numerical and categorical features from channels."""
        features = {}

        for channel in channels:
            channel_id = str(channel.id)
            description = getattr(channel, "description", "") or ""

            features[channel_id] = {
                "subscriber_count": getattr(channel, "subscriber_count", 0) or 0,
                "video_count": getattr(channel, "video_count", 0) or 0,
                "description_length": len(description),
                "has_description": 1 if description else 0,
                "language": getattr(channel, "language", "") or "unknown",
                "country": getattr(channel, "country", "") or "unknown",
                "channel_age_days": self._calculate_channel_age(channel),
            }

        return features

    def _extract_channel_tags(
        self, db: Session, channels: List[Channel]
    ) -> Dict[str, List[str]]:
        """Extract tags associated with each channel."""
        channel_tags = {}

        for channel in channels:
            channel_id = str(channel.id)
            tags = []

            # Get channel tags from database
            channel_tag_records = (
                db.query(ChannelTag).filter(ChannelTag.channel_id == channel.id).all()
            )

            for tag_record in channel_tag_records:
                # Get the actual tag content
                content_tag = (
                    db.query(ContentTag)
                    .filter(ContentTag.id == tag_record.tag_id)
                    .first()
                )

                if content_tag:
                    tags.append(str(content_tag.name))

            # Also include channel description for content analysis
            description = getattr(channel, "description", "") or ""
            if description:
                # Simple keyword extraction from description
                description_words = description.lower().split()
                # Filter for meaningful words (longer than 3 characters)
                meaningful_words = [word for word in description_words if len(word) > 3]
                tags.extend(meaningful_words[:10])  # Limit to 10 description keywords

            channel_tags[channel_id] = tags

        return channel_tags

    def _create_tag_vectors(self, channels: List[Channel]):
        """Create TF-IDF vectors from channel tags and descriptions."""
        # Prepare documents for TF-IDF
        documents = []

        for channel in channels:
            channel_id = str(channel.id)
            # Combine tags and description for each channel
            tags = self.channel_tags.get(channel_id, [])
            description = getattr(channel, "description", "") or ""

            # Create document text
            document = " ".join(tags) + " " + description
            documents.append(document)

        # Create TF-IDF vectors
        if len(documents) > 0:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words="english",
                ngram_range=(1, 2),  # Include bigrams
            )

            self.tag_vectors = self.tfidf_vectorizer.fit_transform(documents)

    def _create_feature_vectors(self):
        """Create normalized feature vectors from channel features."""
        if not self.channel_features:
            return

        # Extract numerical features
        feature_matrix = []
        feature_names = [
            "subscriber_count",
            "video_count",
            "description_length",
            "has_description",
            "channel_age_days",
        ]

        for channel_id in self.channels_list:
            features = self.channel_features[channel_id]
            feature_row = [features.get(name, 0) for name in feature_names]
            feature_matrix.append(feature_row)

        # Normalize features
        if len(feature_matrix) > 0:
            self.feature_scaler = StandardScaler()
            self.feature_vectors = self.feature_scaler.fit_transform(
                np.array(feature_matrix)
            )

    def _calculate_channel_similarity(self):
        """Calculate similarity matrix combining tags and features."""
        if self.tag_vectors is None or self.feature_vectors is None:
            return

        # Calculate tag similarity
        tag_similarity = cosine_similarity(self.tag_vectors)

        # Calculate feature similarity
        feature_similarity = cosine_similarity(self.feature_vectors)

        # Combine similarities (weighted average)
        tag_weight = 0.7
        feature_weight = 0.3

        self.channel_similarity_matrix = (
            tag_weight * tag_similarity + feature_weight * feature_similarity
        )

    def _calculate_channel_age(self, channel: Channel) -> int:
        """Calculate channel age in days."""
        created_date = getattr(channel, "created_date", None)
        if created_date:
            from datetime import datetime

            age = datetime.now() - created_date
            return age.days
        return 0

    def get_similar_channels(
        self, channel_id: str, n_recommendations: int = 10, min_similarity: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Get channels similar to a given channel.

        Args:
            channel_id: Target channel ID
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (channel_id, similarity_score) tuples
        """
        if (
            self.channel_similarity_matrix is None
            or channel_id not in self.channels_list
        ):
            return []

        try:
            channel_idx = self.channels_list.index(channel_id)
            similarities = self.channel_similarity_matrix[channel_idx]

            # Get similar channels
            similar_channels = []
            for i, similarity in enumerate(similarities):
                if i != channel_idx and similarity > min_similarity:
                    similar_channel_id = self.channels_list[i]
                    similar_channels.append((similar_channel_id, float(similarity)))

            # Sort by similarity and return top N
            similar_channels.sort(key=lambda x: x[1], reverse=True)
            return similar_channels[:n_recommendations]

        except Exception as e:
            print(f"Error getting similar channels: {e}")
            return []

    def get_user_recommendations(
        self, db: Session, user_id: str, n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for a user based on their subscriptions.

        Args:
            db: Database session
            user_id: Target user ID
            n_recommendations: Number of recommendations to return

        Returns:
            List of (channel_id, score) tuples
        """
        if self.channel_similarity_matrix is None:
            return []

        try:
            # Get user's subscriptions
            subscriptions = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .all()
            )

            if not subscriptions:
                return []

            subscribed_channels = [str(sub.channel_id) for sub in subscriptions]

            # Calculate recommendation scores based on similarity to subscribed channels
            channel_scores = {}

            for subscribed_channel in subscribed_channels:
                similar_channels = self.get_similar_channels(
                    subscribed_channel, n_recommendations * 2, min_similarity=0.05
                )

                for similar_channel_id, similarity in similar_channels:
                    # Skip if user already subscribed
                    if similar_channel_id in subscribed_channels:
                        continue

                    # Accumulate similarity scores
                    if similar_channel_id in channel_scores:
                        channel_scores[similar_channel_id] += similarity
                    else:
                        channel_scores[similar_channel_id] = similarity

            # Normalize scores by number of subscribed channels
            if subscribed_channels:
                for channel_id in channel_scores:
                    channel_scores[channel_id] /= len(subscribed_channels)

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                channel_scores.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error getting user recommendations: {e}")
            return []

    def get_preference_based_recommendations(
        self, db: Session, user_id: str, n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations based on user's preference tags.

        Args:
            db: Database session
            user_id: Target user ID
            n_recommendations: Number of recommendations to return

        Returns:
            List of (channel_id, score) tuples
        """
        try:
            # Get user preference tags
            preference_tags = (
                db.query(UserPreferenceTags)
                .filter(UserPreferenceTags.user_id == user_id)
                .all()
            )

            if not preference_tags:
                return []

            # Create preference profile
            user_preferences = {}
            for pref_tag in preference_tags:
                tag_name = str(getattr(pref_tag, "tag", ""))
                tag_weight = float(getattr(pref_tag, "weight", 0.0))
                user_preferences[tag_name] = tag_weight

            # Get user's existing subscriptions to exclude
            subscriptions = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .all()
            )
            subscribed_channels = {str(sub.channel_id) for sub in subscriptions}

            # Score channels based on preference match
            channel_scores = {}

            for channel_id in self.channels_list:
                if channel_id in subscribed_channels:
                    continue

                channel_tags = self.channel_tags.get(channel_id, [])

                # Calculate preference match score
                score = 0.0
                for tag in channel_tags:
                    if tag.lower() in user_preferences:
                        score += user_preferences[tag.lower()]

                if score > 0:
                    channel_scores[channel_id] = score

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                channel_scores.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error getting preference-based recommendations: {e}")
            return []

    def get_category_recommendations(
        self, db: Session, user_id: str, category: str, n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations within a specific category.

        Args:
            db: Database session
            user_id: Target user ID
            category: Category to filter by
            n_recommendations: Number of recommendations to return

        Returns:
            List of (channel_id, score) tuples
        """
        try:
            # Get channels in the specified category
            category_channels = []

            for channel_id in self.channels_list:
                channel_tags = self.channel_tags.get(channel_id, [])
                if category.lower() in [tag.lower() for tag in channel_tags]:
                    category_channels.append(channel_id)

            if not category_channels:
                return []

            # Get user's subscriptions to find preferences within category
            subscriptions = (
                db.query(UserSubscription)
                .filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.is_active.is_(True),
                )
                .all()
            )

            subscribed_channels = {str(sub.channel_id) for sub in subscriptions}
            subscribed_in_category = [
                ch for ch in category_channels if ch in subscribed_channels
            ]

            if not subscribed_in_category:
                # No subscriptions in category, return random channels from category
                import random

                random.shuffle(category_channels)
                return [(ch, 1.0) for ch in category_channels[:n_recommendations]]

            # Get recommendations based on similarity to subscribed channels in category
            channel_scores = {}

            for subscribed_channel in subscribed_in_category:
                similar_channels = self.get_similar_channels(
                    subscribed_channel, n_recommendations * 2, min_similarity=0.1
                )

                for similar_channel_id, similarity in similar_channels:
                    # Only include channels in the target category
                    if (
                        similar_channel_id in category_channels
                        and similar_channel_id not in subscribed_channels
                    ):
                        if similar_channel_id in channel_scores:
                            channel_scores[similar_channel_id] += similarity
                        else:
                            channel_scores[similar_channel_id] = similarity

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                channel_scores.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error getting category recommendations: {e}")
            return []

    def get_channel_features_summary(self, channel_id: str) -> Dict[str, Any]:
        """
        Get a summary of features for a specific channel.

        Args:
            channel_id: Channel ID to analyze

        Returns:
            Dictionary with channel feature summary
        """
        if channel_id not in self.channels_list:
            return {}

        try:
            channel_features = self.channel_features.get(channel_id, {})
            channel_tags = self.channel_tags.get(channel_id, [])

            # Get similar channels
            similar_channels = self.get_similar_channels(channel_id, 5)

            return {
                "channel_id": channel_id,
                "features": channel_features,
                "tags": channel_tags,
                "tag_count": len(channel_tags),
                "similar_channels": similar_channels,
                "content_categories": list(
                    set([tag for tag in channel_tags if len(tag) > 3])
                )[:10],
            }

        except Exception as e:
            print(f"Error getting channel features summary: {e}")
            return {}

    def is_trained(self) -> bool:
        """Check if the content-based recommender is trained."""
        return (
            self.channel_similarity_matrix is not None
            and self.tag_vectors is not None
            and self.feature_vectors is not None
        )

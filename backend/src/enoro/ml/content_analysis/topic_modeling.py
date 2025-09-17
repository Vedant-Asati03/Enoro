"""
Topic modeling and clustering service for content analysis.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.tags import (
    ChannelCluster,
    TopicCluster,
)
from backend.src.enoro.ml.content_analysis.feature_extraction import FeatureExtractor


@dataclass
class TopicInfo:
    """Information about a discovered topic."""

    topic_id: int
    name: str
    keywords: List[str]
    channels: List[str]
    coherence_score: float
    channel_count: int


@dataclass
class UserProfile:
    """User interest profile based on subscriptions."""

    user_id: str
    primary_interests: List[str]
    interest_scores: Dict[str, float]
    cluster_membership: str
    subscription_patterns: Dict[str, Any]


class TopicModeler:
    """
    Topic modeling service for discovering content themes and user interests.

    Uses LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization)
    to identify topics in channel descriptions and group similar content.
    """

    def __init__(self, n_topics: int = 20):
        """
        Initialize topic modeler.

        Args:
            n_topics: Number of topics to discover
        """
        self.n_topics = n_topics

        # Models
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method="online",
            max_iter=20,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )

        self.nmf_model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200,
            alpha_W=0.1,
            alpha_H=0.1,
        )

        # Vectorizer for text processing
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Clustering models
        self.channel_clusterer = KMeans(n_clusters=10, random_state=42)
        self.user_clusterer = KMeans(n_clusters=8, random_state=42)

        # Model state
        self._is_fitted = False
        self.topic_labels = {}
        self.topic_keywords = {}

    def extract_channel_texts(self, db: Session) -> List[Tuple[str, str]]:
        """
        Extract text data from all channels.

        Args:
            db: Database session

        Returns:
            List of (channel_id, text) tuples
        """
        channels = db.query(Channel).all()
        texts = []

        for channel in channels:
            # Combine channel name and description
            text_parts = []

            if channel.name:
                text_parts.append(channel.name)

            if channel.description:
                text_parts.append(channel.description)

            combined_text = " ".join(text_parts)

            if len(combined_text.strip()) > 10:  # Minimum text length
                texts.append((channel.id, combined_text))

        return texts

    def fit_topic_models(self, db: Session) -> Dict[str, TopicInfo]:
        """
        Fit topic models on channel data.

        Args:
            db: Database session

        Returns:
            Dictionary of discovered topics
        """
        # Extract text data
        channel_texts = self.extract_channel_texts(db)

        if len(channel_texts) < 10:
            raise ValueError(
                "Insufficient data for topic modeling (need at least 10 channels)"
            )

        channel_ids = [cid for cid, _ in channel_texts]
        texts = [text for _, text in channel_texts]

        # Preprocess texts
        processed_texts = [
            self.feature_extractor.preprocess_text(text) for text in texts
        ]

        # Vectorize
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)

        # Fit LDA model
        self.lda_model.fit(tfidf_matrix)

        # Fit NMF model
        self.nmf_model.fit(tfidf_matrix)

        # Extract topic information
        topics = self._extract_topic_info(channel_ids, tfidf_matrix)

        self._is_fitted = True

        return topics

    def _extract_topic_info(
        self, channel_ids: List[str], tfidf_matrix
    ) -> Dict[str, TopicInfo]:
        """
        Extract topic information from fitted models.

        Args:
            channel_ids: List of channel IDs
            tfidf_matrix: TF-IDF matrix

        Returns:
            Dictionary of topic information
        """
        topics = {}
        feature_names = self.vectorizer.get_feature_names_out()

        # Get topic distributions
        lda_doc_topics = self.lda_model.transform(tfidf_matrix)

        for topic_idx in range(self.n_topics):
            # Get top keywords for this topic
            topic_words = self.lda_model.components_[topic_idx]
            top_word_indices = topic_words.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_word_indices]

            # Find channels strongly associated with this topic
            topic_channels = []
            for doc_idx, doc_topics in enumerate(lda_doc_topics):
                if doc_topics[topic_idx] > 0.3:  # Strong association threshold
                    topic_channels.append(channel_ids[doc_idx])

            # Generate topic name from keywords
            topic_name = self._generate_topic_name(keywords)

            # Calculate coherence (simplified)
            coherence = self._calculate_topic_coherence(keywords, topic_words)

            topics[f"topic_{topic_idx}"] = TopicInfo(
                topic_id=topic_idx,
                name=topic_name,
                keywords=keywords,
                channels=topic_channels,
                coherence_score=coherence,
                channel_count=len(topic_channels),
            )

        return topics

    def _generate_topic_name(self, keywords: List[str]) -> str:
        """
        Generate a human-readable topic name from keywords.

        Args:
            keywords: List of topic keywords

        Returns:
            Generated topic name
        """
        # Predefined topic mappings
        topic_mappings = {
            "tech": ["technology", "programming", "software", "coding", "development"],
            "gaming": ["game", "gaming", "gameplay", "stream", "esports"],
            "education": ["tutorial", "learn", "education", "course", "teach"],
            "entertainment": ["funny", "comedy", "entertainment", "fun"],
            "music": ["music", "song", "artist", "album", "audio"],
            "lifestyle": ["vlog", "lifestyle", "daily", "life"],
            "business": ["business", "entrepreneur", "finance", "money"],
            "health": ["fitness", "health", "workout", "nutrition"],
            "science": ["science", "research", "physics", "chemistry"],
            "art": ["art", "drawing", "creative", "design"],
        }

        # Check if keywords match any predefined categories
        for category, category_keywords in topic_mappings.items():
            if any(keyword in keywords for keyword in category_keywords):
                return category.title()

        # Fallback: use most prominent keyword
        return keywords[0].title() if keywords else "Unknown"

    def _calculate_topic_coherence(
        self, keywords: List[str], topic_weights: np.ndarray
    ) -> float:
        """
        Calculate topic coherence score (simplified).

        Args:
            keywords: Topic keywords
            topic_weights: Topic word weights

        Returns:
            Coherence score (0-1)
        """
        # Simplified coherence based on weight distribution
        if len(topic_weights) == 0:
            return 0.0

        # Calculate entropy (lower entropy = more coherent)
        normalized_weights = topic_weights / topic_weights.sum()
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))

        # Normalize to 0-1 scale (higher = more coherent)
        max_entropy = np.log(len(normalized_weights))
        coherence = 1.0 - (entropy / max_entropy)

        return max(0.0, min(1.0, coherence))

    def cluster_channels(self, db: Session) -> Dict[str, List[str]]:
        """
        Cluster channels based on content similarity.

        Args:
            db: Database session

        Returns:
            Dictionary of cluster_name -> [channel_ids]
        """
        if not self._is_fitted:
            raise ValueError("Models must be fitted before clustering")

        # Get channel vectors
        channel_texts = self.extract_channel_texts(db)

        if len(channel_texts) < 5:
            return {}

        channel_ids = [cid for cid, _ in channel_texts]
        texts = [text for _, text in channel_texts]

        # Vectorize
        processed_texts = [
            self.feature_extractor.preprocess_text(text) for text in texts
        ]
        tfidf_matrix = self.vectorizer.transform(processed_texts)

        # Cluster
        n_clusters = min(10, len(channel_ids) // 2)  # Adaptive cluster count
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(tfidf_matrix.toarray())

        # Group channels by cluster
        clusters = defaultdict(list)
        for channel_id, cluster_id in zip(channel_ids, cluster_labels):
            clusters[f"cluster_{cluster_id}"].append(channel_id)

        return dict(clusters)

    def analyze_user_interests(self, db: Session, user_id: str) -> UserProfile:
        """
        Analyze user interests based on subscription patterns.

        Args:
            db: Database session
            user_id: User to analyze

        Returns:
            User interest profile
        """
        # Get user subscriptions
        subscriptions = (
            db.query(UserSubscription).filter(UserSubscription.user_id == user_id).all()
        )

        if not subscriptions:
            return UserProfile(
                user_id=user_id,
                primary_interests=[],
                interest_scores={},
                cluster_membership="unknown",
                subscription_patterns={},
            )

        # Get subscribed channels
        channel_ids = [sub.channel_id for sub in subscriptions]
        channels = db.query(Channel).filter(Channel.id.in_(channel_ids)).all()

        # Analyze content of subscribed channels
        interest_scores = defaultdict(float)
        all_keywords = []

        for channel in channels:
            if channel.description:
                # Extract content categories
                analysis = self.feature_extractor.analyze_channel_content(channel)

                # Accumulate interest scores
                for topic in analysis.content_categories:
                    interest_scores[topic] += 1.0

                all_keywords.extend(analysis.keywords[:5])  # Top 5 keywords per channel

        # Normalize interest scores
        total_channels = len(channels)
        if total_channels > 0:
            for topic in interest_scores:
                interest_scores[topic] /= total_channels

        # Get primary interests (top 5)
        sorted_interests = sorted(
            interest_scores.items(), key=lambda x: x[1], reverse=True
        )
        primary_interests = [topic for topic, score in sorted_interests[:5]]

        # Determine user cluster (simplified)
        cluster_membership = self._determine_user_cluster(primary_interests)

        # Subscription patterns
        patterns = {
            "total_subscriptions": len(subscriptions),
            "top_keywords": Counter(all_keywords).most_common(10),
            "subscription_diversity": len(set(interest_scores.keys())),
        }

        return UserProfile(
            user_id=user_id,
            primary_interests=primary_interests,
            interest_scores=dict(interest_scores),
            cluster_membership=cluster_membership,
            subscription_patterns=patterns,
        )

    def _determine_user_cluster(self, interests: List[str]) -> str:
        """
        Determine user cluster based on interests.

        Args:
            interests: List of user interests

        Returns:
            Cluster name
        """
        if not interests:
            return "general"

        # Define user archetypes
        archetypes = {
            "tech_enthusiast": ["technology", "programming", "software"],
            "gamer": ["gaming", "entertainment"],
            "learner": ["education", "science"],
            "creator": ["art", "music", "lifestyle"],
            "professional": ["business", "health"],
            "generalist": [],  # Default fallback
        }

        # Score each archetype
        archetype_scores = {}
        for archetype, keywords in archetypes.items():
            score = sum(1 for interest in interests if interest in keywords)
            archetype_scores[archetype] = score

        # Return best matching archetype
        best_archetype = max(archetype_scores.items(), key=lambda x: x[1])

        return best_archetype[0] if best_archetype[1] > 0 else "generalist"

    def save_topics_to_db(self, db: Session, topics: Dict[str, TopicInfo]) -> None:
        """
        Save discovered topics to database.

        Args:
            db: Database session
            topics: Dictionary of topics to save
        """
        for topic_key, topic_info in topics.items():
            # Create or update topic cluster
            cluster = (
                db.query(TopicCluster)
                .filter(TopicCluster.name == topic_info.name)
                .first()
            )

            if not cluster:
                cluster = TopicCluster(
                    name=topic_info.name,
                    description=f"Auto-generated topic: {', '.join(topic_info.keywords[:5])}",
                    keywords=topic_info.keywords,
                    channel_count=topic_info.channel_count,
                    coherence_score=topic_info.coherence_score,
                    model_version="1.0",
                )
                db.add(cluster)
                db.flush()  # Get the ID
            else:
                # Update existing
                cluster.keywords = topic_info.keywords
                cluster.channel_count = topic_info.channel_count
                cluster.coherence_score = topic_info.coherence_score

            # Save channel associations
            for channel_id in topic_info.channels:
                existing_association = (
                    db.query(ChannelCluster)
                    .filter(
                        ChannelCluster.channel_id == channel_id,
                        ChannelCluster.cluster_id == cluster.id,
                    )
                    .first()
                )

                if not existing_association:
                    association = ChannelCluster(
                        channel_id=channel_id,
                        cluster_id=cluster.id,
                        probability=0.8,  # Simplified probability
                        dominant_cluster=True,
                    )
                    db.add(association)

        db.commit()

    def generate_recommendations(
        self, db: Session, user_id: str, limit: int = 10
    ) -> List[str]:
        """
        Generate channel recommendations for a user.

        Args:
            db: Database session
            user_id: User to generate recommendations for
            limit: Maximum number of recommendations

        Returns:
            List of recommended channel IDs
        """
        # Analyze user interests
        user_profile = self.analyze_user_interests(db, user_id)

        if not user_profile.primary_interests:
            return []

        # Get user's current subscriptions
        current_subs = (
            db.query(UserSubscription.channel_id)
            .filter(UserSubscription.user_id == user_id)
            .all()
        )
        subscribed_channels = {sub.channel_id for sub in current_subs}

        # Find channels in similar topic clusters
        recommendations = []

        for interest in user_profile.primary_interests:
            # Find topic clusters for this interest
            clusters = (
                db.query(TopicCluster)
                .filter(TopicCluster.name.ilike(f"%{interest}%"))
                .all()
            )

            for cluster in clusters:
                # Get channels in this cluster
                channel_associations = (
                    db.query(ChannelCluster)
                    .filter(ChannelCluster.cluster_id == cluster.id)
                    .limit(5)  # Top 5 channels per cluster
                    .all()
                )

                for assoc in channel_associations:
                    if (
                        assoc.channel_id not in subscribed_channels
                        and assoc.channel_id not in recommendations
                    ):
                        recommendations.append(assoc.channel_id)

                        if len(recommendations) >= limit:
                            return recommendations

        return recommendations[:limit]

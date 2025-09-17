"""
Collaborative Filtering Recommendation System.

Implements user-based and item-based collaborative filtering algorithms
to generate channel recommendations based on user subscription patterns.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from backend.src.enoro.ml.shared.data_preprocessing import DataPreprocessor


class CollaborativeFilter:
    """
    Collaborative Filtering recommendation system.

    Supports both user-based and item-based collaborative filtering
    with matrix factorization techniques for scalability.
    """

    def __init__(self, min_subscriptions: int = 3, n_components: int = 50):
        """
        Initialize collaborative filtering system.

        Args:
            min_subscriptions: Minimum subscriptions required for a user to be considered
            n_components: Number of components for matrix factorization
        """
        self.min_subscriptions = min_subscriptions
        self.n_components = n_components
        self.preprocessor = DataPreprocessor()

        # Cached matrices and models
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}

    def fit(self, db: Session) -> bool:
        """
        Train the collaborative filtering models.

        Args:
            db: Database session

        Returns:
            True if training was successful
        """
        try:
            # Get user-item subscription matrix
            matrix, users, items = self.preprocessor.get_user_subscription_matrix(db)

            if matrix.empty or len(users) < 2 or len(items) < 2:
                print("Insufficient data for collaborative filtering")
                return False

            # Filter active users
            active_users = self.preprocessor.filter_active_users(
                db, self.min_subscriptions
            )
            matrix = matrix.loc[matrix.index.intersection(active_users)]

            if matrix.empty:
                print("No active users found for collaborative filtering")
                return False

            self.user_item_matrix = matrix

            # Create user and item mappings for efficient lookups
            self.user_mapping = {user: idx for idx, user in enumerate(matrix.index)}
            self.item_mapping = {item: idx for idx, item in enumerate(matrix.columns)}
            self.reverse_user_mapping = {
                idx: user for user, idx in self.user_mapping.items()
            }
            self.reverse_item_mapping = {
                idx: item for item, idx in self.item_mapping.items()
            }

            # Calculate user similarity matrix
            self._calculate_user_similarity()

            # Calculate item similarity matrix
            self._calculate_item_similarity()

            # Train SVD model for matrix factorization
            self._train_matrix_factorization()

            print(
                f"Collaborative filtering trained on {len(matrix.index)} users and {len(matrix.columns)} channels"
            )
            return True

        except Exception as e:
            print(f"Error training collaborative filtering: {e}")
            return False

    def _calculate_user_similarity(self):
        """Calculate user-user similarity matrix using cosine similarity."""
        if self.user_item_matrix is None:
            return

        # Calculate cosine similarity between users
        user_vectors = self.user_item_matrix.values
        self.user_similarity_matrix = cosine_similarity(user_vectors)

    def _calculate_item_similarity(self):
        """Calculate item-item similarity matrix using cosine similarity."""
        if self.user_item_matrix is None:
            return

        # Calculate cosine similarity between items (channels)
        item_vectors = self.user_item_matrix.T.values
        self.item_similarity_matrix = cosine_similarity(item_vectors)

    def _train_matrix_factorization(self):
        """Train SVD model for matrix factorization recommendations."""
        if self.user_item_matrix is None:
            return

        try:
            # Use TruncatedSVD for dimensionality reduction
            n_components = min(self.n_components, min(self.user_item_matrix.shape) - 1)

            if n_components > 0:
                self.svd_model = TruncatedSVD(
                    n_components=n_components, random_state=42
                )
                self.svd_model.fit(self.user_item_matrix)

        except Exception as e:
            print(f"Error training matrix factorization: {e}")
            self.svd_model = None

    def get_user_based_recommendations(
        self, user_id: str, n_recommendations: int = 10, min_similarity: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations using user-based collaborative filtering.

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold for considering users

        Returns:
            List of (channel_id, score) tuples
        """
        if (
            self.user_item_matrix is None
            or self.user_similarity_matrix is None
            or user_id not in self.user_mapping
        ):
            return []

        try:
            user_idx = self.user_mapping[user_id]
            user_similarities = self.user_similarity_matrix[user_idx]
            user_subscriptions = self.user_item_matrix.iloc[user_idx]

            # Find similar users
            similar_users = []
            for i, similarity in enumerate(user_similarities):
                if i != user_idx and similarity > min_similarity:
                    similar_users.append((i, similarity))

            if not similar_users:
                return []

            # Calculate recommendation scores
            channel_scores = {}

            for channel_idx, channel_id in enumerate(self.user_item_matrix.columns):
                # Skip if user already subscribed
                if user_subscriptions.iloc[channel_idx] > 0:
                    continue

                score = 0.0
                total_similarity = 0.0

                for similar_user_idx, similarity in similar_users:
                    similar_user_subscriptions = self.user_item_matrix.iloc[
                        similar_user_idx
                    ]

                    if similar_user_subscriptions.iloc[channel_idx] > 0:
                        score += (
                            similarity * similar_user_subscriptions.iloc[channel_idx]
                        )
                        total_similarity += similarity

                if total_similarity > 0:
                    normalized_score = score / total_similarity
                    channel_scores[channel_id] = normalized_score

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                channel_scores.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error in user-based recommendations: {e}")
            return []

    def get_item_based_recommendations(
        self, user_id: str, n_recommendations: int = 10, min_similarity: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations using item-based collaborative filtering.

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold for considering items

        Returns:
            List of (channel_id, score) tuples
        """
        if (
            self.user_item_matrix is None
            or self.item_similarity_matrix is None
            or user_id not in self.user_mapping
        ):
            return []

        try:
            user_idx = self.user_mapping[user_id]
            user_subscriptions = self.user_item_matrix.iloc[user_idx]

            # Get user's subscribed channels
            subscribed_channels = []
            for channel_idx, subscription in enumerate(user_subscriptions):
                if subscription > 0:
                    subscribed_channels.append(channel_idx)

            if not subscribed_channels:
                return []

            # Calculate recommendation scores based on item similarities
            channel_scores = {}

            for channel_idx, channel_id in enumerate(self.user_item_matrix.columns):
                # Skip if user already subscribed
                if user_subscriptions.iloc[channel_idx] > 0:
                    continue

                score = 0.0
                total_similarity = 0.0

                for subscribed_channel_idx in subscribed_channels:
                    similarity = self.item_similarity_matrix[channel_idx][
                        subscribed_channel_idx
                    ]

                    if similarity > min_similarity:
                        score += (
                            similarity * user_subscriptions.iloc[subscribed_channel_idx]
                        )
                        total_similarity += similarity

                if total_similarity > 0:
                    normalized_score = score / total_similarity
                    channel_scores[channel_id] = normalized_score

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                channel_scores.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error in item-based recommendations: {e}")
            return []

    def get_matrix_factorization_recommendations(
        self, user_id: str, n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations using matrix factorization (SVD).

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return

        Returns:
            List of (channel_id, score) tuples
        """
        if (
            self.user_item_matrix is None
            or self.svd_model is None
            or user_id not in self.user_mapping
        ):
            return []

        try:
            user_idx = self.user_mapping[user_id]
            user_subscriptions = self.user_item_matrix.iloc[user_idx]

            # Transform user vector using SVD
            user_vector = np.array(user_subscriptions.values).reshape(1, -1)
            user_transformed = self.svd_model.transform(user_vector)

            # Reconstruct full user vector from reduced representation
            reconstructed = self.svd_model.inverse_transform(user_transformed)
            predicted_scores = reconstructed[0]

            # Get recommendations for unsubscribed channels
            recommendations = []

            for channel_idx, channel_id in enumerate(self.user_item_matrix.columns):
                # Skip if user already subscribed
                if user_subscriptions.iloc[channel_idx] > 0:
                    continue

                predicted_score = predicted_scores[channel_idx]
                recommendations.append((channel_id, predicted_score))

            # Sort and return top recommendations
            sorted_recommendations = sorted(
                recommendations, key=lambda x: x[1], reverse=True
            )

            return sorted_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error in matrix factorization recommendations: {e}")
            return []

    def get_combined_recommendations(
        self,
        user_id: str,
        n_recommendations: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations by combining multiple collaborative filtering approaches.

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            weights: Weights for different methods {'user': 0.4, 'item': 0.4, 'matrix': 0.2}

        Returns:
            List of (channel_id, score) tuples
        """
        if weights is None:
            weights = {"user": 0.4, "item": 0.4, "matrix": 0.2}

        # Get recommendations from each method
        user_based = self.get_user_based_recommendations(user_id, n_recommendations * 2)
        item_based = self.get_item_based_recommendations(user_id, n_recommendations * 2)
        matrix_based = self.get_matrix_factorization_recommendations(
            user_id, n_recommendations * 2
        )

        # Combine scores
        combined_scores = {}

        # Add user-based scores
        for channel_id, score in user_based:
            combined_scores[channel_id] = (
                combined_scores.get(channel_id, 0) + weights["user"] * score
            )

        # Add item-based scores
        for channel_id, score in item_based:
            combined_scores[channel_id] = (
                combined_scores.get(channel_id, 0) + weights["item"] * score
            )

        # Add matrix factorization scores
        for channel_id, score in matrix_based:
            combined_scores[channel_id] = (
                combined_scores.get(channel_id, 0) + weights["matrix"] * score
            )

        # Sort and return top recommendations
        sorted_recommendations = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_recommendations[:n_recommendations]

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user for recommendation debugging.

        Args:
            user_id: Target user ID

        Returns:
            Dictionary with user statistics
        """
        if self.user_item_matrix is None or user_id not in self.user_mapping:
            return {}

        user_idx = self.user_mapping[user_id]
        user_subscriptions = self.user_item_matrix.iloc[user_idx]

        stats = {
            "user_id": user_id,
            "total_subscriptions": int(user_subscriptions.sum()),
            "subscription_channels": user_subscriptions[
                user_subscriptions > 0
            ].index.tolist(),
            "similar_users_count": 0,
            "avg_user_similarity": 0.0,
        }

        if self.user_similarity_matrix is not None:
            similarities = self.user_similarity_matrix[user_idx]
            similarities = similarities[similarities != 1.0]  # Exclude self-similarity

            stats["similar_users_count"] = int(np.sum(similarities > 0.1))
            stats["avg_user_similarity"] = (
                float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            )

        return stats

    def is_trained(self) -> bool:
        """Check if the collaborative filtering model is trained."""
        return (
            self.user_item_matrix is not None
            and self.user_similarity_matrix is not None
            and self.item_similarity_matrix is not None
        )

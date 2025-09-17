"""
Data preprocessing utilities for ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.video import Video


class DataPreprocessor:
    """
    Shared data preprocessing utilities for ML pipelines.

    Provides common data transformation and preparation methods
    used across different ML models.
    """

    def __init__(self):
        """Initialize the data preprocessor."""
        pass

    def get_user_subscription_matrix(
        self, db: Session
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Create user-channel subscription matrix for collaborative filtering.

        Args:
            db: Database session

        Returns:
            Tuple of (matrix DataFrame, user_ids, channel_ids)
        """
        # Get all subscriptions
        subscriptions = db.query(UserSubscription).all()

        if not subscriptions:
            return pd.DataFrame(), [], []

        # Create subscription data
        data = []
        for sub in subscriptions:
            data.append(
                {
                    "user_id": sub.user_id,
                    "channel_id": sub.channel_id,
                    "subscribed": 1,  # Binary subscription indicator
                }
            )

        df = pd.DataFrame(data)

        # Create pivot table (user-channel matrix)
        matrix = df.pivot_table(
            index="user_id", columns="channel_id", values="subscribed", fill_value=0
        )

        return matrix, list(matrix.index), list(matrix.columns)

    def get_channel_features_matrix(self, db: Session) -> pd.DataFrame:
        """
        Create channel features matrix for content-based recommendations.

        Args:
            db: Database session

        Returns:
            DataFrame with channel features
        """
        channels = db.query(Channel).all()

        features = []
        for channel in channels:
            # Extract basic features
            feature_row = {
                "channel_id": channel.id,
                "subscriber_count": channel.subscriber_count or 0,
                "video_count": channel.video_count or 0,
                "description_length": len(channel.description or ""),
                "has_description": 1 if channel.description else 0,
                "language": channel.language or "unknown",
                "country": channel.country or "unknown",
            }

            features.append(feature_row)

        return pd.DataFrame(features)

    def normalize_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize specified columns using min-max normalization.

        Args:
            df: Input DataFrame
            columns: Columns to normalize

        Returns:
            DataFrame with normalized columns
        """
        df_normalized = df.copy()

        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[col] = 0

        return df_normalized

    def encode_categorical_features(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features.

        Args:
            df: Input DataFrame
            columns: Categorical columns to encode

        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()

        for col in columns:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)

        return df_encoded

    def filter_active_users(self, db: Session, min_subscriptions: int = 3) -> List[str]:
        """
        Get list of active users with minimum subscription count.

        Args:
            db: Database session
            min_subscriptions: Minimum number of subscriptions required

        Returns:
            List of active user IDs
        """
        # Count subscriptions per user
        user_counts = (
            db.query(UserSubscription.user_id)
            .filter(UserSubscription.is_active == True)
            .group_by(UserSubscription.user_id)
            .having(db.func.count(UserSubscription.channel_id) >= min_subscriptions)
            .all()
        )

        return [user.user_id for user in user_counts]

    def get_user_interaction_history(self, db: Session, user_id: str) -> pd.DataFrame:
        """
        Get user's interaction history for recommendation training.

        Args:
            db: Database session
            user_id: User to get history for

        Returns:
            DataFrame with interaction history
        """
        # Get user subscriptions
        subscriptions = (
            db.query(UserSubscription).filter(UserSubscription.user_id == user_id).all()
        )

        interactions = []
        for sub in subscriptions:
            interactions.append(
                {
                    "user_id": user_id,
                    "channel_id": sub.channel_id,
                    "interaction_type": "subscription",
                    "timestamp": sub.fetched_at,
                    "implicit_rating": 1.0,  # Subscription implies positive preference
                }
            )

        return pd.DataFrame(interactions)

    def create_train_test_split(
        self, matrix: pd.DataFrame, test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split user-item matrix into train and test sets.

        Args:
            matrix: User-item interaction matrix
            test_ratio: Proportion of data for testing

        Returns:
            Tuple of (train_matrix, test_matrix)
        """
        train_matrix = matrix.copy()
        test_matrix = matrix.copy()

        # For each user, randomly select test_ratio of their interactions for testing
        for user_id in matrix.index:
            user_interactions = matrix.loc[user_id]
            positive_interactions = user_interactions[user_interactions > 0].index

            if len(positive_interactions) > 1:
                n_test = max(1, int(len(positive_interactions) * test_ratio))
                test_items = np.random.choice(
                    positive_interactions, n_test, replace=False
                )

                # Set test items to 0 in training matrix
                train_matrix.loc[user_id, test_items] = 0

                # Set all items except test items to 0 in test matrix
                test_matrix.loc[user_id, :] = 0
                test_matrix.loc[user_id, test_items] = matrix.loc[user_id, test_items]

        return train_matrix, test_matrix

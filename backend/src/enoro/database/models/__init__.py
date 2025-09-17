"""
Database models for Enoro.
"""

from backend.src.enoro.database.models.video import (
    Video,
    VideoFeatures,
    UserVideoInteraction,
    UserRating,
)
from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.search import (
    SearchQuery,
    UserPreferenceTags,
    UserProfile,
)
from backend.src.enoro.database.models.tags import (
    ContentTag,
    ChannelTag,
    VideoTag,
    UserInterest,
    TopicCluster,
    ChannelCluster,
    UserCluster,
)
from backend.src.enoro.database.models.base import get_db, create_tables

__all__ = [
    "Video",
    "VideoFeatures",
    "UserVideoInteraction",
    "UserRating",
    "Channel",
    "UserSubscription",
    "SearchQuery",
    "UserPreferenceTags",
    "UserProfile",
    "ContentTag",
    "ChannelTag",
    "VideoTag",
    "UserInterest",
    "TopicCluster",
    "ChannelCluster",
    "UserCluster",
    "get_db",
    "create_tables",
]

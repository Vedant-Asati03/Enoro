"""
Search and user preference related database models.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
)
from sqlalchemy.sql import func

from backend.src.enoro.database.models.base import Base


class SearchQuery(Base):
    """Track user's YouTube search history."""

    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default_user")
    query = Column(String, nullable=False)

    # Search metadata from YouTube
    searched_at = Column(DateTime)  # When user searched on YouTube
    result_count = Column(Integer)  # Number of results returned
    clicked_video_id = Column(String)  # If user clicked on a specific video

    # Our tracking
    fetched_at = Column(DateTime, default=func.now())  # When we fetched this search
    processed = Column(
        Boolean, default=False
    )  # If we've processed this for tag generation

    created_at = Column(DateTime, default=func.now())


class UserPreferenceTags(Base):
    """Store user preference tags generated from YouTube data and evolved through usage."""

    __tablename__ = "user_preference_tags"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default_user")
    tag = Column(String, nullable=False)
    weight = Column(Float, default=1.0)  # Importance/frequency weight
    source = Column(String)  # 'subscription', 'search', 'watched', 'evolved'

    # Tag evolution tracking
    original_weight = Column(Float, default=1.0)  # Initial weight when created
    boost_count = Column(Integer, default=0)  # How many times this tag was boosted
    decay_applied = Column(Boolean, default=False)  # If time-based decay was applied

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class UserProfile(Base):
    """User profile and preferences."""

    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, index=True, default="default_user")

    # YouTube account info
    youtube_user_id = Column(String)  # User's YouTube account ID
    youtube_email = Column(String)  # Email associated with YouTube account
    youtube_name = Column(String)  # Display name from YouTube

    # Preferences
    preferred_categories = Column(Text)  # JSON string
    preferred_languages = Column(Text)  # JSON string
    preferred_duration_min = Column(Integer)
    preferred_duration_max = Column(Integer)

    # Learning style preferences
    prefers_tutorials = Column(Boolean, default=True)
    prefers_short_content = Column(Boolean, default=False)
    prefers_project_based = Column(Boolean, default=True)
    difficulty_level = Column(
        String, default="intermediate"
    )  # beginner, intermediate, advanced

    # Data sync tracking
    last_subscription_sync = Column(DateTime)  # Last time we synced subscriptions
    last_search_sync = Column(DateTime)  # Last time we synced search history
    last_tag_generation = Column(DateTime)  # Last time we generated preference tags

    # Activity tracking
    total_videos_rated = Column(Integer, default=0)
    total_videos_liked = Column(Integer, default=0)
    total_watch_time = Column(Integer, default=0)  # seconds
    last_active = Column(DateTime, default=func.now())

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

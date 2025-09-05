"""
Video-related database models.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    BigInteger,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.src.enoro.database.models.base import Base


class Video(Base):
    """Video model."""

    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    channel_name = Column(String, nullable=False)
    channel_id = Column(String, nullable=False)
    thumbnail_url = Column(String)
    duration = Column(String)
    duration_seconds = Column(Integer)  # Duration in seconds for calculations
    published_at = Column(DateTime)
    view_count = Column(BigInteger, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    category_id = Column(Integer)
    category_name = Column(String)
    tags = Column(Text)  # Original YouTube tags (JSON string)
    language = Column(String, default="en")
    url = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    features = relationship("VideoFeatures", back_populates="video", uselist=False)
    ratings = relationship("UserRating", back_populates="video")
    interactions = relationship("UserVideoInteraction", back_populates="video")
    tags = relationship("VideoTag", back_populates="video")


class VideoFeatures(Base):
    """Video features for ML."""

    __tablename__ = "video_features"

    video_id = Column(String, ForeignKey("videos.id"), primary_key=True)

    # Content features
    title_length = Column(Integer)
    description_length = Column(Integer)
    tags_count = Column(Integer, default=0)

    # Engagement features
    view_like_ratio = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    popularity_score = Column(Float, default=0.0)

    # Sentiment and keyword features
    title_sentiment = Column(Float, default=0.0)
    description_sentiment = Column(Float, default=0.0)

    # Boolean keyword features
    has_tutorial_keywords = Column(Boolean, default=False)
    has_beginner_keywords = Column(Boolean, default=False)
    has_advanced_keywords = Column(Boolean, default=False)
    has_time_constraint = Column(Boolean, default=False)
    has_project_keywords = Column(Boolean, default=False)
    has_trending_keywords = Column(Boolean, default=False)

    # Category-specific features
    category_popularity = Column(Float, default=0.0)
    channel_subscriber_count = Column(Integer, default=0)
    channel_video_count = Column(Integer, default=0)

    # Temporal features
    days_since_published = Column(Integer, default=0)
    upload_frequency_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    video = relationship("Video", back_populates="features")


class UserRating(Base):
    """User ratings for videos."""

    __tablename__ = "user_ratings"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    user_id = Column(String, default="default_user")  # For future multi-user support

    # Rating data
    liked = Column(Boolean, nullable=False)
    rating_score = Column(Float)  # 1-5 scale, optional
    watch_time = Column(Integer)  # seconds watched
    completion_rate = Column(Float)  # percentage watched

    # User feedback
    notes = Column(Text)
    feedback_tags = Column(Text)  # JSON string for structured feedback

    # Implicit feedback
    clicked = Column(Boolean, default=True)
    shared = Column(Boolean, default=False)
    bookmarked = Column(Boolean, default=False)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    video = relationship("Video", back_populates="ratings")


class UserVideoInteraction(Base):
    """Track user video interactions for recommendations."""

    __tablename__ = "user_video_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default_user")
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    channel_id = Column(String, nullable=False)  # Just store as string, no FK

    # Interaction data
    watched_at = Column(DateTime, default=func.now())
    watch_duration_seconds = Column(Integer)
    completion_percentage = Column(Float)
    rating = Column(Integer)  # 1-5 stars, optional

    created_at = Column(DateTime, default=func.now())

    # Relationships
    video = relationship("Video", back_populates="interactions")

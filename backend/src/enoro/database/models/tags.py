"""
ML-generated tags and content analysis models.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.src.enoro.database.models.base import Base


class ContentTag(Base):
    """Auto-generated tags for content categorization."""

    __tablename__ = "content_tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    category = Column(String, nullable=False)  # 'topic', 'genre', 'style', etc.
    description = Column(Text)
    confidence_score = Column(Float, default=0.0)  # ML confidence in this tag

    # Tag metadata
    usage_count = Column(Integer, default=0)  # How many times this tag is used
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    channel_tags = relationship("ChannelTag", back_populates="tag")
    video_tags = relationship("VideoTag", back_populates="tag")


class ChannelTag(Base):
    """Tags assigned to YouTube channels."""

    __tablename__ = "channel_tags"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("content_tags.id"), nullable=False)

    # ML scores
    relevance_score = Column(Float, nullable=False)  # How relevant this tag is
    confidence_score = Column(Float, nullable=False)  # ML confidence
    source = Column(String, nullable=False)  # 'description', 'titles', 'collaborative'

    # Analysis metadata
    analysis_version = Column(String, default="1.0")  # Track ML model versions

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    tag = relationship("ContentTag", back_populates="channel_tags")
    channel = relationship("Channel", back_populates="tags")


class VideoTag(Base):
    """Tags assigned to YouTube videos."""

    __tablename__ = "video_tags"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("content_tags.id"), nullable=False)

    # ML scores
    relevance_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    source = Column(String, nullable=False)  # 'title', 'description', 'thumbnail'

    created_at = Column(DateTime, default=func.now())

    # Relationships
    tag = relationship("ContentTag", back_populates="video_tags")
    video = relationship("Video", back_populates="tags")


class UserInterest(Base):
    """ML-generated user interest profiles."""

    __tablename__ = "user_interests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    tag_id = Column(Integer, ForeignKey("content_tags.id"), nullable=False)

    # Interest scores
    interest_score = Column(Float, nullable=False)  # 0.0 to 1.0
    engagement_score = Column(Float, default=0.0)  # Based on subscription patterns

    # Analysis metadata
    calculated_from = Column(JSON)  # Track which subscriptions influenced this
    analysis_date = Column(DateTime, default=func.now())

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    tag = relationship("ContentTag")


class TopicCluster(Base):
    """Content topic clusters identified by ML."""

    __tablename__ = "topic_clusters"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    # Cluster metadata
    keywords = Column(JSON)  # Top keywords for this cluster
    channel_count = Column(Integer, default=0)  # Channels in this cluster
    user_count = Column(Integer, default=0)  # Users interested in this cluster

    # ML metadata
    model_version = Column(String, default="1.0")
    coherence_score = Column(Float)  # Topic coherence metric

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ChannelCluster(Base):
    """Channels assigned to topic clusters."""

    __tablename__ = "channel_clusters"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("topic_clusters.id"), nullable=False)

    # Assignment scores
    probability = Column(Float, nullable=False)  # Probability of belonging to cluster
    dominant_cluster = Column(Boolean, default=False)  # Primary cluster for channel

    created_at = Column(DateTime, default=func.now())

    # Relationships
    channel = relationship("Channel")
    cluster = relationship("TopicCluster")


class UserCluster(Base):
    """Users grouped by content consumption patterns."""

    __tablename__ = "user_clusters"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    cluster_name = Column(
        String, nullable=False
    )  # 'tech_enthusiast', 'gaming_fan', etc.

    # Cluster scores
    membership_score = Column(Float, nullable=False)  # How well user fits cluster
    primary_cluster = Column(Boolean, default=False)

    # Cluster characteristics
    interests = Column(JSON)  # Primary interests for this cluster
    subscription_patterns = Column(JSON)  # Common subscription patterns

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

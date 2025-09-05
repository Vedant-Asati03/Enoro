"""
Channel-related database models.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    BigInteger,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.src.enoro.database.models.base import Base


class Channel(Base):
    """Channel information from YouTube."""

    __tablename__ = "channels"

    id = Column(String, primary_key=True, index=True)  # YouTube channel ID
    name = Column(String, nullable=False)
    description = Column(Text)
    subscriber_count = Column(BigInteger, default=0)
    video_count = Column(Integer, default=0)
    channel_url = Column(String)
    thumbnail_url = Column(String)

    # Channel metadata
    country = Column(String)
    language = Column(String)
    created_date = Column(DateTime)  # When channel was created on YouTube

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    subscriptions = relationship("UserSubscription", back_populates="channel")
    tags = relationship("ChannelTag", back_populates="channel")


class UserSubscription(Base):
    """Track user's YouTube subscriptions."""

    __tablename__ = "user_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default_user")
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)

    # Subscription metadata from YouTube
    subscribed_at = Column(DateTime)  # When user subscribed on YouTube
    notification_level = Column(String)  # 'all', 'personalized', 'none'

    # Our tracking
    fetched_at = Column(
        DateTime, default=func.now()
    )  # When we fetched this subscription
    is_active = Column(Boolean, default=True)  # If subscription is still active

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    channel = relationship("Channel", back_populates="subscriptions")

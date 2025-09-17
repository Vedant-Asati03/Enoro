"""
YouTube Data API v3 service for fetching user data.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session

from backend.src.enoro.core.config import settings
from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.search import UserProfile

logger = logging.getLogger(__name__)


class YouTubeAPIService:
    """Service for interacting with YouTube Data API v3."""

    def __init__(self):
        """Initialize YouTube API service."""
        self.api_service_name = "youtube"
        self.api_version = "v3"

    def _build_service(self, credentials_dict: Dict[str, Any]):
        """
        Build YouTube API service with credentials.

        Args:
            credentials_dict: Dictionary containing OAuth credentials

        Returns:
            YouTube API service instance
        """
        try:
            credentials = Credentials(
                token=credentials_dict.get("access_token"),
                refresh_token=credentials_dict.get("refresh_token"),
                token_uri=credentials_dict.get("token_uri"),
                client_id=credentials_dict.get("client_id"),
                client_secret=credentials_dict.get("client_secret"),
                scopes=credentials_dict.get("scopes"),
            )

            return build(
                self.api_service_name, self.api_version, credentials=credentials
            )

        except Exception as e:
            logger.error(f"Error building YouTube service: {e}")
            raise

    def fetch_user_subscriptions(
        self,
        credentials_dict: Dict[str, Any],
        db: Session,
        user_id: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Fetch user's YouTube subscriptions and store in database.

        Args:
            credentials_dict: OAuth credentials
            db: Database session
            user_id: User identifier

        Returns:
            Dict with fetch statistics
        """
        try:
            youtube = self._build_service(credentials_dict)

            subscriptions_count = 0
            new_channels_count = 0
            next_page_token = None

            while True:
                # Fetch subscriptions page
                request = youtube.subscriptions().list(
                    part="snippet,contentDetails",
                    mine=True,
                    maxResults=50,
                    pageToken=next_page_token,
                )

                response = request.execute()

                for item in response.get("items", []):
                    try:
                        subscription_data = self._process_subscription_item(
                            item, db, user_id
                        )
                        if subscription_data["is_new_channel"]:
                            new_channels_count += 1
                        subscriptions_count += 1

                    except Exception as e:
                        logger.warning(f"Error processing subscription item: {e}")
                        continue

                # Check for next page
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            # Update user profile sync timestamp
            self._update_user_sync_timestamp(db, user_id, "subscription")

            db.commit()

            result = {
                "total_subscriptions": subscriptions_count,
                "new_channels": new_channels_count,
                "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Fetched {subscriptions_count} subscriptions, {new_channels_count} new channels"
            )
            return result

        except HttpError as e:
            logger.error(f"YouTube API error fetching subscriptions: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching subscriptions: {e}")
            raise

    def _process_subscription_item(
        self, item: Dict[str, Any], db: Session, user_id: str
    ) -> Dict[str, Any]:
        """
        Process a single subscription item and store in database.

        Args:
            item: Subscription item from YouTube API
            db: Database session
            user_id: User identifier

        Returns:
            Dict with processing information
        """
        snippet = item.get("snippet", {})
        resource_id = snippet.get("resourceId", {})

        channel_id = resource_id.get("channelId")
        if not channel_id:
            raise ValueError("No channel ID found in subscription item")

        # Check if channel exists, create if not
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        is_new_channel = False

        if not channel:
            # Fetch channel details
            channel_data = self._fetch_channel_details(channel_id, db)
            if channel_data:
                channel = Channel(**channel_data)
                db.add(channel)
                is_new_channel = True

        # Check if subscription already exists
        existing_subscription = (
            db.query(UserSubscription)
            .filter(
                UserSubscription.user_id == user_id,
                UserSubscription.channel_id == channel_id,
            )
            .first()
        )

        if not existing_subscription:
            # Create new subscription record
            subscription = UserSubscription(
                user_id=user_id,
                channel_id=channel_id,
                subscribed_at=self._parse_youtube_date(snippet.get("publishedAt")),
                is_active=True,
                fetched_at=datetime.now(timezone.utc),
            )
            db.add(subscription)
        else:
            # Update existing subscription
            existing_subscription.is_active = True
            existing_subscription.fetched_at = datetime.now(timezone.utc)

        return {
            "channel_id": channel_id,
            "is_new_channel": is_new_channel,
            "subscription_exists": existing_subscription is not None,
        }

    def _fetch_channel_details(
        self, channel_id: str, db: Session
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed channel information from YouTube API.

        Args:
            channel_id: YouTube channel ID
            db: Database session

        Returns:
            Dict with channel data or None if error
        """
        try:
            # Use API key for public channel data
            youtube = build(
                self.api_service_name,
                self.api_version,
                developerKey=settings.youtube_api_key,
            )

            request = youtube.channels().list(part="snippet,statistics", id=channel_id)

            response = request.execute()
            items = response.get("items", [])

            if not items:
                logger.warning(f"Channel {channel_id} not found")
                return None

            item = items[0]
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            return {
                "id": channel_id,
                "name": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "subscriber_count": int(statistics.get("subscriberCount", 0)),
                "video_count": int(statistics.get("videoCount", 0)),
                "channel_url": f"https://www.youtube.com/channel/{channel_id}",
                "thumbnail_url": snippet.get("thumbnails", {})
                .get("default", {})
                .get("url"),
                "country": snippet.get("country"),
                "language": snippet.get("defaultLanguage"),
                "created_date": self._parse_youtube_date(snippet.get("publishedAt")),
            }

        except HttpError as e:
            logger.error(f"YouTube API error fetching channel {channel_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching channel {channel_id}: {e}")
            return None

    def fetch_user_search_activity(
        self,
        credentials_dict: Dict[str, Any],
        db: Session,
        user_id: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Attempt to fetch user's search activity.
        Note: YouTube API doesn't provide direct access to search history.
        This would require YouTube Analytics API or other methods.

        Args:
            credentials_dict: OAuth credentials
            db: Database session
            user_id: User identifier

        Returns:
            Dict with status information
        """
        logger.info(
            "Search history fetching not available via standard YouTube Data API"
        )

        # Update user profile sync timestamp anyway
        self._update_user_sync_timestamp(db, user_id, "search")
        db.commit()

        return {
            "status": "search_history_not_available",
            "message": "YouTube Data API v3 doesn't provide access to user search history",
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _update_user_sync_timestamp(self, db: Session, user_id: str, sync_type: str):
        """
        Update user profile sync timestamps.

        Args:
            db: Database session
            user_id: User identifier
            sync_type: Type of sync ('subscription' or 'search')
        """
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()

        if not user_profile:
            user_profile = UserProfile(id=user_id)
            db.add(user_profile)

        now = datetime.now(timezone.utc)

        if sync_type == "subscription":
            user_profile.last_subscription_sync = now
        elif sync_type == "search":
            user_profile.last_search_sync = now

        user_profile.last_active = now

    def _parse_youtube_date(self, date_string: Optional[str]) -> Optional[datetime]:
        """
        Parse YouTube API date string to datetime.

        Args:
            date_string: ISO format date string from YouTube API

        Returns:
            Parsed datetime or None
        """
        if not date_string:
            return None

        try:
            # YouTube API returns ISO format dates
            return datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        except Exception as e:
            logger.warning(f"Error parsing date {date_string}: {e}")
            return None

    def get_quota_usage_estimate(self, operation: str) -> int:
        """
        Get estimated quota usage for different operations.

        Args:
            operation: Operation type

        Returns:
            Estimated quota cost
        """
        quota_costs = {
            "subscriptions_list": 1,
            "channels_list": 1,
            "search": 100,
            "videos_list": 1,
        }

        return quota_costs.get(operation, 1)


# Global YouTube API service instance
youtube_api = YouTubeAPIService()

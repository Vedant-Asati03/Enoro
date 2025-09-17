"""
Caching and quota management service for staying within free tier limits.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from sqlalchemy.orm import Session

from backend.src.enoro.database.models.search import UserProfile

logger = logging.getLogger(__name__)


class QuotaManager:
    """Manages API quota usage and caching to stay within free tier limits."""

    # Free tier limits
    DAILY_QUOTA_LIMIT = 10000  # YouTube Data API v3 daily quota
    SYNC_COOLDOWN_DAYS = 7  # Cache subscriptions for 7 days
    MAX_DAILY_NEW_USERS = 100  # Beta limit for new registrations

    # API costs (YouTube Data API v3 quota units)
    COSTS = {
        "subscriptions_list": 1,  # Per 50 subscriptions
        "channels_list": 1,  # Per channel info request
        "search": 100,  # Per search request (expensive!)
        "videos_list": 1,  # Per video details request
    }

    def should_sync_subscriptions(self, user_profile: UserProfile) -> bool:
        """
        Check if user's subscriptions should be synced based on cache policy.

        Args:
            user_profile: User profile with sync timestamps

        Returns:
            True if sync is needed, False if cached data is still fresh
        """
        if user_profile.last_subscription_sync is None:
            logger.info(f"First time sync for user {user_profile.id}")
            return True

        last_sync = user_profile.last_subscription_sync
        if last_sync.tzinfo is None:
            last_sync = last_sync.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        days_since_sync = (now - last_sync).days

        if days_since_sync >= self.SYNC_COOLDOWN_DAYS:
            logger.info(
                f"Sync needed for user {user_profile.id}: {days_since_sync} days since last sync"
            )
            return True
        else:
            logger.info(
                f"Sync skipped for user {user_profile.id}: only {days_since_sync} days since last sync"
            )
            return False

    def update_sync_timestamp(self, user_profile: UserProfile, db: Session) -> None:
        """
        Update the user's last subscription sync timestamp.

        Args:
            user_profile: User profile to update
            db: Database session
        """
        # Update the user profile instance
        db.query(UserProfile).filter(UserProfile.id == user_profile.id).update(
            {"last_subscription_sync": datetime.now(timezone.utc)}
        )
        db.commit()
        logger.info(f"Updated sync timestamp for user {user_profile.id}")

    def estimate_sync_cost(self, subscription_count: int) -> int:
        """
        Estimate API quota cost for syncing user subscriptions.

        Args:
            subscription_count: Number of subscriptions to sync

        Returns:
            Estimated quota units needed
        """
        # Subscriptions list API: 1 unit per 50 subscriptions
        subscription_cost = max(1, (subscription_count + 49) // 50)

        # Channel details: 1 unit per channel (but we batch these)
        channel_cost = max(1, (subscription_count + 49) // 50)

        total_cost = subscription_cost + channel_cost
        logger.debug(
            f"Estimated cost for {subscription_count} subscriptions: {total_cost} quota units"
        )
        return total_cost

    def can_afford_sync(self, estimated_cost: int, daily_usage: int = 0) -> bool:
        """
        Check if we can afford a sync operation within daily quota.

        Args:
            estimated_cost: Quota units needed for operation
            daily_usage: Current daily quota usage

        Returns:
            True if operation is within budget
        """
        remaining_quota = self.DAILY_QUOTA_LIMIT - daily_usage
        can_afford = estimated_cost <= remaining_quota

        if not can_afford:
            logger.warning(
                f"Quota insufficient: need {estimated_cost}, have {remaining_quota}"
            )

        return can_afford

    def get_cache_status(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Get cache status information for a user.

        Args:
            user_profile: User profile to check

        Returns:
            Cache status information
        """
        if user_profile.last_subscription_sync is None:
            return {
                "cached": False,
                "last_sync": None,
                "days_since_sync": None,
                "next_sync_due": "now",
                "cache_valid": False,
            }

        last_sync = user_profile.last_subscription_sync
        if last_sync.tzinfo is None:
            last_sync = last_sync.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        days_since_sync = (now - last_sync).days
        next_sync_date = last_sync + timedelta(days=self.SYNC_COOLDOWN_DAYS)

        return {
            "cached": True,
            "last_sync": last_sync.isoformat(),
            "days_since_sync": days_since_sync,
            "next_sync_due": next_sync_date.isoformat(),
            "cache_valid": days_since_sync < self.SYNC_COOLDOWN_DAYS,
        }


class BetaUserLimiter:
    """Manages beta user registration limits."""

    def __init__(self):
        self.daily_limit = QuotaManager.MAX_DAILY_NEW_USERS
        self._daily_count = 0
        self._last_reset = datetime.now(timezone.utc).date()

    def can_register_new_user(self, db: Session) -> bool:
        """
        Check if a new user can register within beta limits.

        Args:
            db: Database session

        Returns:
            True if registration is allowed
        """
        self._reset_daily_count_if_needed()

        if self._daily_count >= self.daily_limit:
            logger.warning(
                f"Daily registration limit reached: {self._daily_count}/{self.daily_limit}"
            )
            return False

        return True

    def record_new_user(self) -> None:
        """Record a new user registration."""
        self._reset_daily_count_if_needed()
        self._daily_count += 1
        logger.info(
            f"New user registered: {self._daily_count}/{self.daily_limit} today"
        )

    def _reset_daily_count_if_needed(self) -> None:
        """Reset daily count if it's a new day."""
        today = datetime.now(timezone.utc).date()
        if today > self._last_reset:
            self._daily_count = 0
            self._last_reset = today
            logger.info("Daily registration count reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current beta user limit status."""
        self._reset_daily_count_if_needed()

        return {
            "daily_limit": self.daily_limit,
            "registrations_today": self._daily_count,
            "remaining_slots": self.daily_limit - self._daily_count,
            "accepting_new_users": self._daily_count < self.daily_limit,
        }


# Global instances
quota_manager = QuotaManager()
beta_limiter = BetaUserLimiter()

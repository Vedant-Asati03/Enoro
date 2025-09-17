"""
YouTube authentication and sync API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from backend.src.enoro.database.database import get_db
from backend.src.enoro.services.youtube import youtube_oauth, youtube_api
from backend.src.enoro.services.session_manager import session_manager
from backend.src.enoro.database.models.search import UserProfile
from backend.src.enoro.api.middleware.auth import require_auth, get_current_user
from backend.src.enoro.api.auth.errors import (
    handle_youtube_api_error,
    log_authentication_event,
)
from backend.src.enoro.services.quota_manager import quota_manager, beta_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/youtube", tags=["youtube-auth"])


@router.get("/login")
async def youtube_login(request: Request) -> Dict[str, str]:
    """
    Initiate YouTube OAuth login flow.

    Returns:
        Dict with authorization URL
    """
    try:
        # Generate state for CSRF protection
        state = f"user_session_{id(request)}"

        authorization_url, flow_state = youtube_oauth.get_authorization_url(state)

        # In production, you should store the state in session/cache
        # For now, we'll just return it

        return {
            "authorization_url": authorization_url,
            "state": flow_state,
            "message": "Redirect user to authorization_url to complete OAuth flow",
        }

    except Exception as e:
        logger.error(f"Error initiating YouTube login: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate YouTube login")


@router.get("/callback")
async def youtube_callback(
    code: str = Query(..., description="Authorization code from YouTube"),
    state: str = Query(..., description="State parameter for CSRF protection"),
    error: str = Query(None, description="Error from OAuth provider"),
    db: Session = Depends(get_db),
):
    """
    Handle YouTube OAuth callback.

    Args:
        code: Authorization code from YouTube
        state: State parameter for CSRF protection
        error: Error parameter if OAuth failed
        db: Database session

    Returns:
        Redirect to success page or error response
    """
    try:
        # Check for OAuth errors
        if error:
            log_authentication_event(
                "oauth_callback_error", details=error, success=False
            )
            logger.warning(f"OAuth error: {error}")
            return RedirectResponse(
                url=f"http://localhost:3000/auth/error?error={error}", status_code=302
            )

        if not code:
            log_authentication_event("oauth_callback_missing_code", success=False)
            return RedirectResponse(
                url="http://localhost:3000/auth/error?error=missing_code",
                status_code=302,
            )

        # Exchange code for tokens
        try:
            token_data = youtube_oauth.exchange_code_for_tokens(code, state)
            user_info = token_data.get("user_info", {})
            user_id = user_info.get("id", "unknown")
            user_name = user_info.get("snippet", {}).get("title", "Unknown User")

            log_authentication_event(
                "oauth_token_exchange", user_id, user_name, success=True
            )

        except Exception as token_error:
            log_authentication_event(
                "oauth_token_exchange_failed", details=str(token_error), success=False
            )
            logger.error(f"Token exchange failed: {token_error}")
            return RedirectResponse(
                url="http://localhost:3000/auth/error?error=token_exchange_failed",
                status_code=302,
            )

        # Create user session with encrypted token storage
        try:
            session_manager.create_session(user_info, token_data)
            log_authentication_event(
                "session_created", user_id, user_name, success=True
            )
        except Exception as session_error:
            log_authentication_event(
                "session_creation_failed",
                user_id,
                user_name,
                str(session_error),
                success=False,
            )
            logger.error(f"Session creation failed: {session_error}")
            return RedirectResponse(
                url="http://localhost:3000/auth/error?error=session_creation_failed",
                status_code=302,
            )

        # Create/update user profile in database
        user_profile = (
            db.query(UserProfile)
            .filter(UserProfile.youtube_user_id == user_info.get("id"))
            .first()
        )

        is_new_user = user_profile is None

        # Check beta user limits for new users
        if is_new_user and not beta_limiter.can_register_new_user(db):
            log_authentication_event(
                "registration_blocked_beta_limit",
                user_id,
                user_name,
                "Daily registration limit reached",
                success=False,
            )
            return RedirectResponse(
                url="http://localhost:3000/auth/error?error=beta_limit_reached&message=Registration limit reached for today. Please try again tomorrow.",
                status_code=302,
            )

        if not user_profile:
            user_profile = UserProfile(
                id=user_info.get("id", "default_user"),
                youtube_user_id=user_info.get("id"),
                youtube_name=user_info.get("snippet", {}).get("title"),
            )
            db.add(user_profile)
            # Record new user registration
            beta_limiter.record_new_user()
            log_authentication_event(
                "new_user_registered", user_id, user_name, success=True
            )
        else:
            # Update existing profile
            user_profile.youtube_name = user_info.get("snippet", {}).get("title")

        db.commit()

        # Automatic subscription synchronization after successful authentication
        user_id = user_info.get("id", "default_user")
        user_name = user_info.get("snippet", {}).get("title", "Unknown")

        try:
            # Fetch and store subscriptions automatically
            credentials_dict = token_data
            result = youtube_api.fetch_user_subscriptions(credentials_dict, db, user_id)

            logger.info(
                f"Automatic subscription sync completed for {user_name}: "
                f"{result.get('subscriptions_found', 0)} subscriptions processed"
            )

        except Exception as sync_error:
            # Don't fail the authentication if sync fails
            logger.warning(
                f"Automatic subscription sync failed for {user_name}: {sync_error}"
            )

        logger.info(f"YouTube OAuth successful for channel: {user_name}")

        # Redirect to success page
        return RedirectResponse(
            url="http://localhost:3000/auth/success", status_code=302
        )

    except Exception as e:
        logger.error(f"Error handling YouTube callback: {e}")
        return RedirectResponse(
            url="http://localhost:3000/auth/error?error=authentication_failed",
            status_code=302,
        )


@router.post("/sync/subscriptions")
async def sync_subscriptions(
    db: Session = Depends(get_db),
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Sync user's YouTube subscriptions using session-based authentication.

    This endpoint uses the authentication middleware to automatically validate
    user sessions and extract credentials securely.

    Args:
        db: Database session
        user_context: User authentication context from middleware

    Returns:
        Dict with sync results
    """
    try:
        # Get user info from authenticated session
        user_info = user_context["user"]

        # Extract user ID from user info
        user_id = user_info.get("id", "default_user")
        user_name = user_info.get("snippet", {}).get("title", "Unknown User")

        log_authentication_event("subscription_sync_started", user_id, user_name)

        # Get user profile to check cache status
        user_profile = (
            db.query(UserProfile).filter(UserProfile.youtube_user_id == user_id).first()
        )

        if not user_profile:
            raise HTTPException(
                status_code=404,
                detail="User profile not found. Please re-authenticate.",
            )

        # Check if sync is needed based on caching policy
        if not quota_manager.should_sync_subscriptions(user_profile):
            cache_status = quota_manager.get_cache_status(user_profile)
            log_authentication_event(
                "subscription_sync_skipped_cached",
                user_id,
                user_name,
                f"Cache valid, {cache_status['days_since_sync']} days old",
            )
            return {
                "status": "cached",
                "message": f"Subscriptions are cached (last sync: {cache_status['days_since_sync']} days ago). Next sync available in {7 - cache_status['days_since_sync']} days.",
                "user": {
                    "id": user_id,
                    "name": user_name,
                },
                "cache_info": cache_status,
            }

        # Build credentials dict for the YouTube API service
        session_data = session_manager.get_session()
        if not session_data or not session_data.get("tokens"):
            log_authentication_event(
                "subscription_sync_invalid_session", user_id, user_name, success=False
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid session. Please re-authenticate.",
            )

        credentials_dict = session_data["tokens"]

        # Fetch and store subscriptions
        try:
            result = youtube_api.fetch_user_subscriptions(credentials_dict, db, user_id)

            # Update sync timestamp
            quota_manager.update_sync_timestamp(user_profile, db)

            log_authentication_event(
                "subscription_sync_completed",
                user_id,
                user_name,
                f"Found {result.get('subscriptions_found', 0)} subscriptions",
            )
        except Exception as api_error:
            # Handle YouTube API specific errors
            youtube_error = handle_youtube_api_error(api_error)
            log_authentication_event(
                "subscription_sync_api_error",
                user_id,
                user_name,
                str(youtube_error),
                success=False,
            )
            raise HTTPException(
                status_code=youtube_error.status_code,
                detail=youtube_error.message,
            )

        logger.info(f"Subscription sync completed for user {user_name} ({user_id})")
        return {
            "status": "success",
            "sync_results": result,
            "user": {
                "id": user_id,
                "name": user_name,
            },
            "message": "Subscriptions synced successfully",
            "cache_info": quota_manager.get_cache_status(user_profile),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing subscriptions: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync subscriptions")


@router.post("/sync/search")
async def sync_search_history(
    db: Session = Depends(get_db),
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Attempt to sync user's YouTube search history using session-based authentication.

    Note: YouTube Data API v3 doesn't provide access to user search history.
    This endpoint is included for completeness and future potential integrations.

    Args:
        db: Database session
        user_context: User authentication context from middleware

    Returns:
        Dict with sync status
    """
    try:
        # Get user info from authenticated session
        user_info = user_context["user"]

        # Extract user ID from user info
        user_id = user_info.get("id", "default_user")
        user_name = user_info.get("snippet", {}).get("title", "Unknown User")

        # Build credentials dict for the YouTube API service
        session_data = session_manager.get_session()
        if not session_data or not session_data.get("tokens"):
            raise HTTPException(
                status_code=401,
                detail="Invalid session. Please re-authenticate.",
            )

        credentials_dict = session_data["tokens"]

        # Attempt to fetch search history (will return not available message)
        result = youtube_api.fetch_user_search_activity(credentials_dict, db, user_id)

        return {
            "status": "info",
            "sync_results": result,
            "user": {
                "id": user_id,
                "name": user_name,
            },
            "message": "Search history sync attempted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing search history: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync search history")


@router.get("/status")
async def youtube_auth_status(
    db: Session = Depends(get_db),
    user_context: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get YouTube authentication and sync status.

    Args:
        db: Database session
        user_context: Optional user authentication context

    Returns:
        Dict with authentication status
    """
    try:
        # Check if user is authenticated
        is_authenticated = user_context is not None
        user_id = None
        user_name = None
        last_subscription_sync = None

        if is_authenticated:
            user_info = user_context["user"]
            user_id = user_info.get("id", "default_user")
            user_name = user_info.get("snippet", {}).get("title", "Unknown User")

            # Check for user profile in database to get sync timestamps
            user_profile = (
                db.query(UserProfile)
                .filter(UserProfile.youtube_user_id == user_id)
                .first()
            )

            if user_profile:
                if (
                    hasattr(user_profile, "updated_at")
                    and user_profile.updated_at is not None
                ):
                    last_subscription_sync = user_profile.updated_at.isoformat()

        return {
            "authenticated": is_authenticated,
            "user": {
                "id": user_id,
                "name": user_name,
            }
            if is_authenticated
            else None,
            "last_subscription_sync": last_subscription_sync,
            "last_search_sync": None,  # Search sync not available via YouTube API
            "available_endpoints": {
                "login": "/api/v1/auth/youtube/login",
                "callback": "/api/v1/auth/youtube/callback",
                "sync_subscriptions": "/api/v1/auth/youtube/sync/subscriptions",
                "sync_search": "/api/v1/auth/youtube/sync/search",
                "status": "/api/v1/auth/youtube/status",
            },
            "quota_info": {
                "subscriptions_cost": youtube_api.get_quota_usage_estimate(
                    "subscriptions_list"
                ),
                "search_cost": youtube_api.get_quota_usage_estimate("search"),
                "daily_limit": 10000,
            },
        }

    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get authentication status"
        )


# Session Management Endpoints


@router.get("/session/user")
async def get_current_user_info(
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Get current authenticated user information.

    Args:
        user_context: User authentication context from middleware

    Returns:
        Dict with current user information
    """
    try:
        user_info = user_context["user"]

        return {
            "status": "success",
            "user": {
                "id": user_info.get("id"),
                "name": user_info.get("snippet", {}).get("title"),
                "description": user_info.get("snippet", {}).get("description"),
                "custom_url": user_info.get("snippet", {}).get("customUrl"),
                "thumbnail_url": user_info.get("snippet", {})
                .get("thumbnails", {})
                .get("default", {})
                .get("url"),
                "country": user_info.get("snippet", {}).get("country"),
                "view_count": user_info.get("statistics", {}).get("viewCount"),
                "subscriber_count": user_info.get("statistics", {}).get(
                    "subscriberCount"
                ),
                "video_count": user_info.get("statistics", {}).get("videoCount"),
                "published_at": user_info.get("snippet", {}).get("publishedAt"),
            },
            "session_info": {
                "session_id": user_context["session_id"],
                "is_authenticated": True,
            },
        }

    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user information")


@router.get("/session/check")
async def check_session_status(
    user_context: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Check if user has an active session (no authentication required).

    Args:
        user_context: Optional user authentication context

    Returns:
        Dict with session status
    """
    try:
        is_authenticated = user_context is not None

        if is_authenticated:
            user_info = user_context["user"]
            return {
                "authenticated": True,
                "user": {
                    "id": user_info.get("id"),
                    "name": user_info.get("snippet", {}).get("title"),
                },
                "session_id": user_context["session_id"],
                "token_expiry": session_manager.is_token_expired(),
            }
        else:
            return {
                "authenticated": False,
                "user": None,
                "session_id": None,
                "token_expiry": True,
            }

    except Exception as e:
        logger.error(f"Error checking session: {e}")
        return {
            "authenticated": False,
            "user": None,
            "session_id": None,
            "error": "Failed to check session status",
        }


@router.post("/session/refresh")
async def refresh_user_session(
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Manually refresh user session tokens.

    Args:
        user_context: User authentication context from middleware

    Returns:
        Dict with refresh status
    """
    try:
        # Get current session data
        session_data = session_manager.get_session()
        if not session_data or not session_data.get("tokens"):
            raise HTTPException(
                status_code=401,
                detail="No valid session found",
            )

        # Get refresh token
        refresh_token = session_data["tokens"].get("refresh_token")
        if not refresh_token:
            raise HTTPException(
                status_code=400,
                detail="No refresh token available",
            )

        # Refresh tokens using YouTube OAuth service
        new_tokens = youtube_oauth.refresh_access_token(refresh_token)

        # Update session with new tokens
        refresh_success = session_manager.refresh_tokens(new_tokens)

        if refresh_success:
            return {
                "status": "success",
                "message": "Session tokens refreshed successfully",
                "user": {
                    "id": user_context["user"].get("id"),
                    "name": user_context["user"].get("snippet", {}).get("title"),
                },
                "token_expiry": session_manager.is_token_expired(),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update session with new tokens",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing session: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh session")


@router.post("/logout")
async def logout_user(
    user_context: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Logout user and clear their session.

    Args:
        user_context: Optional user authentication context

    Returns:
        Dict with logout status
    """
    try:
        if user_context:
            user_info = user_context["user"]
            user_name = user_info.get("snippet", {}).get("title", "Unknown User")

            # Clear the user session
            session_manager.clear_session()

            logger.info(f"User logged out: {user_name}")

            return {
                "status": "success",
                "message": f"Successfully logged out {user_name}",
                "authenticated": False,
            }
        else:
            # No active session to logout
            return {
                "status": "info",
                "message": "No active session found",
                "authenticated": False,
            }

    except Exception as e:
        logger.error(f"Error during logout: {e}")
        # Clear session anyway to ensure logout
        try:
            session_manager.clear_session()
        except Exception:
            pass

        return {
            "status": "success",
            "message": "Logout completed (session cleared due to error)",
            "authenticated": False,
        }


@router.delete("/session")
async def delete_user_session(
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Delete user session (requires authentication).

    Args:
        user_context: User authentication context from middleware

    Returns:
        Dict with deletion status
    """
    try:
        user_info = user_context["user"]
        user_name = user_info.get("snippet", {}).get("title", "Unknown User")

        # Clear the user session
        session_manager.clear_session()

        logger.info(f"User session deleted: {user_name}")

        return {
            "status": "success",
            "message": f"Session deleted for {user_name}",
            "authenticated": False,
        }

    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        # Clear session anyway
        try:
            session_manager.clear_session()
        except Exception:
            pass

        raise HTTPException(status_code=500, detail="Failed to delete session")


# Beta Management Endpoints


@router.get("/beta/status")
async def get_beta_status() -> Dict[str, Any]:
    """
    Get current beta user limits and quota status.

    Returns:
        Dict with beta limits and current usage
    """
    try:
        beta_status = beta_limiter.get_status()

        return {
            "status": "success",
            "beta_limits": {
                "daily_registration_limit": beta_status["daily_limit"],
                "registrations_today": beta_status["registrations_today"],
                "remaining_slots": beta_status["remaining_slots"],
                "accepting_new_users": beta_status["accepting_new_users"],
            },
            "api_quota": {
                "daily_limit": quota_manager.DAILY_QUOTA_LIMIT,
                "cache_duration_days": quota_manager.SYNC_COOLDOWN_DAYS,
                "estimated_cost_per_user": "3-5 quota units",
                "free_tier_user_capacity": "~2,500 users/day",
            },
            "cache_policy": {
                "subscription_sync_frequency": f"Every {quota_manager.SYNC_COOLDOWN_DAYS} days",
                "automatic_sync_on_login": "Only if cache expired",
                "manual_sync_respect_cache": True,
            },
        }

    except Exception as e:
        logger.error(f"Error getting beta status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get beta status")


@router.post("/force-sync")
async def force_subscription_sync(
    db: Session = Depends(get_db),
    user_context: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Force subscription sync regardless of cache (admin/debug endpoint).

    Args:
        db: Database session
        user_context: User authentication context from middleware

    Returns:
        Dict with sync results
    """
    try:
        # Get user info from authenticated session
        user_info = user_context["user"]
        user_id = user_info.get("id", "default_user")
        user_name = user_info.get("snippet", {}).get("title", "Unknown User")

        log_authentication_event("force_sync_started", user_id, user_name)

        # Get session data
        session_data = session_manager.get_session()
        if not session_data or not session_data.get("tokens"):
            raise HTTPException(
                status_code=401,
                detail="Invalid session. Please re-authenticate.",
            )

        credentials_dict = session_data["tokens"]

        # Force sync regardless of cache
        try:
            result = youtube_api.fetch_user_subscriptions(credentials_dict, db, user_id)

            # Update sync timestamp
            user_profile = (
                db.query(UserProfile)
                .filter(UserProfile.youtube_user_id == user_id)
                .first()
            )

            if user_profile:
                quota_manager.update_sync_timestamp(user_profile, db)

            log_authentication_event(
                "force_sync_completed",
                user_id,
                user_name,
                f"Found {result.get('subscriptions_found', 0)} subscriptions",
            )
        except Exception as api_error:
            youtube_error = handle_youtube_api_error(api_error)
            log_authentication_event(
                "force_sync_api_error",
                user_id,
                user_name,
                str(youtube_error),
                success=False,
            )
            raise HTTPException(
                status_code=youtube_error.status_code,
                detail=youtube_error.message,
            )

        logger.info(f"Force sync completed for user {user_name} ({user_id})")
        return {
            "status": "success",
            "sync_results": result,
            "user": {
                "id": user_id,
                "name": user_name,
            },
            "message": "Force sync completed successfully (cache bypassed)",
            "warning": "This endpoint bypasses cache and consumes API quota",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in force sync: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to force sync subscriptions"
        )

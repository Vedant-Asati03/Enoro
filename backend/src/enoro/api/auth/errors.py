"""
Custom exception handlers and error types for authentication flow.
"""

from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
import logging

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Base exception for authentication errors."""

    def __init__(
        self, message: str, error_code: str = "auth_error", status_code: int = 401
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


class SessionError(AuthenticationError):
    """Exception for session-related errors."""

    def __init__(self, message: str, error_code: str = "session_error"):
        super().__init__(message, error_code, 401)


class TokenRefreshError(AuthenticationError):
    """Exception for token refresh errors."""

    def __init__(self, message: str, error_code: str = "token_refresh_error"):
        super().__init__(message, error_code, 401)


class YouTubeAPIError(Exception):
    """Exception for YouTube API errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "youtube_api_error",
        status_code: int = 502,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


def create_error_response(
    error: Exception,
    default_message: str = "An error occurred",
    default_status: int = 500,
) -> Dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error: Exception that occurred
        default_message: Default error message
        default_status: Default HTTP status code

    Returns:
        Standardized error response dictionary
    """
    if isinstance(error, (AuthenticationError, YouTubeAPIError)):
        return {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "type": type(error).__name__,
            },
            "status": "error",
            "authenticated": False,
        }
    elif isinstance(error, HTTPException):
        return {
            "error": {
                "code": "http_error",
                "message": error.detail,
                "type": "HTTPException",
            },
            "status": "error",
            "authenticated": False,
        }
    else:
        logger.error(f"Unexpected error: {error}")
        return {
            "error": {
                "code": "internal_error",
                "message": default_message,
                "type": "InternalError",
            },
            "status": "error",
            "authenticated": False,
        }


def create_error_redirect(
    error: Exception,
    base_url: str = "http://localhost:3000/auth/error",
) -> RedirectResponse:
    """
    Create error redirect response for OAuth flow.

    Args:
        error: Exception that occurred
        base_url: Base URL for error redirects

    Returns:
        RedirectResponse with error parameters
    """
    if isinstance(error, AuthenticationError):
        error_code = error.error_code
        message = error.message
    elif isinstance(error, HTTPException):
        error_code = "http_error"
        message = error.detail
    else:
        error_code = "internal_error"
        message = "Authentication failed"

    return RedirectResponse(
        url=f"{base_url}?error={error_code}&message={message}", status_code=302
    )


def handle_youtube_api_error(error: Exception) -> YouTubeAPIError:
    """
    Convert YouTube API errors to standardized format.

    Args:
        error: Original YouTube API error

    Returns:
        YouTubeAPIError with appropriate message
    """
    error_message = str(error)

    if "quota" in error_message.lower():
        return YouTubeAPIError(
            "YouTube API quota exceeded. Please try again later.", "quota_exceeded", 429
        )
    elif "forbidden" in error_message.lower():
        return YouTubeAPIError(
            "Access forbidden. Please check your YouTube permissions.",
            "access_forbidden",
            403,
        )
    elif "not found" in error_message.lower():
        return YouTubeAPIError(
            "Requested YouTube resource not found.", "resource_not_found", 404
        )
    elif "unauthorized" in error_message.lower():
        return YouTubeAPIError(
            "YouTube authentication required. Please re-authenticate.",
            "unauthorized",
            401,
        )
    else:
        return YouTubeAPIError(f"YouTube API error: {error_message}", "api_error", 502)


def validate_session_data(session_data: Dict[str, Any]) -> None:
    """
    Validate session data structure.

    Args:
        session_data: Session data to validate

    Raises:
        SessionError: If session data is invalid
    """
    if not session_data:
        raise SessionError("Session data is empty")

    if "tokens" not in session_data:
        raise SessionError("Session missing token data")

    tokens = session_data["tokens"]
    if not tokens.get("access_token"):
        raise SessionError("Session missing access token")

    if "user_info" not in session_data:
        raise SessionError("Session missing user information")

    if "expires_at" not in session_data:
        raise SessionError("Session missing expiration data")


def log_authentication_event(
    event_type: str,
    user_id: str | None = None,
    user_name: str | None = None,
    details: str | None = None,
    success: bool = True,
) -> None:
    """
    Log authentication events for monitoring.

    Args:
        event_type: Type of authentication event
        user_id: User ID if available
        user_name: User name if available
        details: Additional details
        success: Whether the event was successful
    """
    log_level = logging.INFO if success else logging.WARNING

    log_data = {
        "event": event_type,
        "success": success,
        "user_id": user_id,
        "user_name": user_name,
        "details": details,
    }

    # Remove None values
    log_data = {k: v for k, v in log_data.items() if v is not None}

    logger.log(log_level, f"Auth event: {log_data}")

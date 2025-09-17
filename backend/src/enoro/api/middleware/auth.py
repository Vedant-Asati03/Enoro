"""
Authentication middleware for automatic user context injection and token validation.
"""

import logging
from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.src.enoro.services.session_manager import session_manager
from backend.src.enoro.services.youtube.oauth import youtube_oauth

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic user authentication and context injection.

    This middleware:
    1. Checks for user sessions in every request
    2. Validates tokens and refreshes if needed
    3. Injects user context into request state
    4. Handles authentication errors gracefully
    """

    def __init__(self, app, exclude_paths: Optional[list] = None):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application instance
            exclude_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/auth/youtube/login",
            "/auth/youtube/callback",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain

        Returns:
            HTTP response
        """
        # Skip authentication for excluded paths
        if self._should_skip_auth(request.url.path):
            return await call_next(request)

        # Initialize user context
        request.state.user = None
        request.state.session_id = None
        request.state.is_authenticated = False

        try:
            # Get session from session manager
            session_data = session_manager.get_session()

            if session_data:
                # Validate and refresh tokens if needed
                validated_session = await self._validate_and_refresh_session(
                    session_data
                )

                if validated_session:
                    # Inject user context into request
                    request.state.user = validated_session.get("user_info")
                    request.state.session_id = validated_session.get("session_id")
                    request.state.is_authenticated = True
                    request.state.access_token = validated_session.get(
                        "tokens", {}
                    ).get("access_token")

                    # Safe logging with user context
                    user_name = "Unknown"
                    if request.state.user and isinstance(request.state.user, dict):
                        user_snippet = request.state.user.get("snippet", {})
                        if isinstance(user_snippet, dict):
                            user_name = user_snippet.get("title", "Unknown")

                    logger.debug(f"User authenticated: {user_name}")
                else:
                    # Session invalid, clear it
                    session_manager.clear_session()
                    logger.warning("Invalid session cleared")

        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            # Continue without authentication rather than failing the request

        return await call_next(request)

    def _should_skip_auth(self, path: str) -> bool:
        """
        Check if authentication should be skipped for this path.

        Args:
            path: Request path

        Returns:
            True if authentication should be skipped
        """
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    async def _validate_and_refresh_session(self, session_data: dict) -> Optional[dict]:
        """
        Validate session and refresh tokens if needed.

        Args:
            session_data: Current session data

        Returns:
            Updated session data or None if invalid
        """
        try:
            tokens = session_data.get("tokens", {})
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")

            if not access_token:
                logger.warning("No access token in session")
                return None

            # Check if token is still valid using the validate_credentials method
            try:
                is_valid = youtube_oauth.validate_credentials(tokens)

                if is_valid:
                    # Token is still valid, return current session
                    return session_data
                else:
                    raise Exception("Token validation failed")

            except Exception as token_error:
                logger.warning(f"Access token invalid: {token_error}")

                # Try to refresh token if we have a refresh token
                if refresh_token:
                    try:
                        new_tokens = youtube_oauth.refresh_access_token(refresh_token)

                        # Update session with new tokens using the session manager
                        refresh_success = session_manager.refresh_tokens(new_tokens)

                        if refresh_success:
                            # Get the updated session data
                            updated_session = session_manager.get_session()
                            logger.info("Successfully refreshed access token")
                            return updated_session
                        else:
                            logger.error(
                                "Failed to update session with refreshed tokens"
                            )
                            return None

                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh token: {refresh_error}")
                        return None
                else:
                    logger.warning("No refresh token available")
                    return None

        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None


def require_auth(request: Request) -> dict:
    """
    Dependency to require authentication for endpoints.

    Args:
        request: HTTP request with user context

    Returns:
        User context from authenticated session

    Raises:
        HTTPException: If user is not authenticated
    """
    if not getattr(request.state, "is_authenticated", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in through /auth/youtube/login",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user": request.state.user,
        "session_id": request.state.session_id,
        "access_token": request.state.access_token,
    }


def get_current_user(request: Request) -> Optional[dict]:
    """
    Dependency to get current user if authenticated (optional).

    Args:
        request: HTTP request with user context

    Returns:
        User context if authenticated, None otherwise
    """
    if getattr(request.state, "is_authenticated", False):
        return {
            "user": request.state.user,
            "session_id": request.state.session_id,
            "access_token": request.state.access_token,
        }
    return None

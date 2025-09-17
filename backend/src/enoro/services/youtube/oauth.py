"""
YouTube OAuth authentication service.
"""

import logging
from typing import Optional, Dict, Any

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.src.enoro.core.config import settings

logger = logging.getLogger(__name__)


class YouTubeOAuthService:
    """Handle YouTube OAuth2 authentication flow."""

    def __init__(self):
        """Initialize OAuth service."""
        self.client_config = {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],
            }
        }
        self.scopes = settings.youtube_scopes

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Generate OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            tuple: (authorization_url, state)
        """
        try:
            flow = Flow.from_client_config(
                client_config=self.client_config,
                scopes=self.scopes,
                redirect_uri=settings.google_redirect_uri,
            )

            # Generate authorization URL
            authorization_url, flow_state = flow.authorization_url(
                access_type="offline",  # Enables refresh token
                include_granted_scopes="true",
                state=state,
                prompt="consent",  # Forces consent to get refresh token
            )

            final_state = flow_state or state or ""
            logger.info(f"Generated OAuth URL for state: {final_state}")
            return str(authorization_url), str(final_state)

        except Exception as e:
            logger.error(f"Error generating authorization URL: {e}")
            raise

    def exchange_code_for_tokens(
        self, authorization_code: str, state: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access tokens.

        Args:
            authorization_code: Authorization code from OAuth callback
            state: State parameter for CSRF protection

        Returns:
            Dict containing token information and user data
        """
        try:
            flow = Flow.from_client_config(
                client_config=self.client_config,
                scopes=self.scopes,
                redirect_uri=settings.google_redirect_uri,
                state=state,
            )

            # Exchange authorization code for tokens
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials

            # Get user info from YouTube
            user_info = self._get_user_info(credentials)

            token_data = {
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": getattr(
                    credentials, "token_uri", self.client_config["web"]["token_uri"]
                ),
                "client_id": getattr(
                    credentials, "client_id", self.client_config["web"]["client_id"]
                ),
                "client_secret": getattr(
                    credentials,
                    "client_secret",
                    self.client_config["web"]["client_secret"],
                ),
                "scopes": list(credentials.scopes)
                if credentials.scopes
                else self.scopes,
                "expiry": credentials.expiry.isoformat()
                if credentials.expiry
                else None,
                "user_info": user_info,
            }

            logger.info(
                f"Successfully exchanged tokens for user: {user_info.get('snippet', {}).get('title', 'Unknown')}"
            )
            return token_data

        except Exception as e:
            logger.error(f"Error exchanging authorization code: {e}")
            raise

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous authorization

        Returns:
            Dict containing new token information
        """
        try:
            credentials = Credentials(
                token=None,
                refresh_token=refresh_token,
                token_uri=self.client_config["web"]["token_uri"],
                client_id=self.client_config["web"]["client_id"],
                client_secret=self.client_config["web"]["client_secret"],
            )

            # Refresh the token
            credentials.refresh(Request())

            token_data = {
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "expiry": credentials.expiry.isoformat()
                if credentials.expiry
                else None,
            }

            logger.info("Successfully refreshed access token")
            return token_data

        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            raise

    def _get_user_info(self, credentials) -> Dict[str, Any]:
        """
        Get user's YouTube channel information.

        Args:
            credentials: OAuth2 credentials

        Returns:
            Dict containing user's channel information
        """
        try:
            youtube = build("youtube", "v3", credentials=credentials)

            # Get user's channel info
            request = youtube.channels().list(part="snippet,statistics", mine=True)
            response = request.execute()

            if response.get("items"):
                return response["items"][0]
            else:
                logger.warning("No channel found for authenticated user")
                return {}

        except HttpError as e:
            logger.error(f"YouTube API error getting user info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            raise

    def validate_credentials(self, credentials_dict: Dict[str, Any]) -> bool:
        """
        Validate if credentials are still valid.

        Args:
            credentials_dict: Dictionary containing credential information

        Returns:
            bool: True if credentials are valid
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

            # Try to make a simple API call
            youtube = build("youtube", "v3", credentials=credentials)
            request = youtube.channels().list(part="id", mine=True)
            request.execute()

            return True

        except Exception as e:
            logger.warning(f"Credentials validation failed: {e}")
            return False


# Global OAuth service instance
youtube_oauth = YouTubeOAuthService()

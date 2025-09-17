"""
User session management with JSON file storage.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet

from backend.src.enoro.core.config import ENORO_CONFIG_DIR

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions with encrypted JSON file storage."""

    def __init__(self):
        """Initialize session manager."""
        self.session_dir = ENORO_CONFIG_DIR
        self.session_file = self.session_dir / "user_session.json"
        self.key_file = self.session_dir / "session_key.key"

        # Ensure directory exists (already handled by config.py but just in case)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for session data."""
        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            return key

    def _encrypt_data(self, data: str) -> str:
        """Encrypt session data."""
        cipher = Fernet(self._get_encryption_key())
        return cipher.encrypt(data.encode()).decode()

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt session data."""
        cipher = Fernet(self._get_encryption_key())
        return cipher.decrypt(encrypted_data.encode()).decode()

    def create_session(
        self, user_data: Dict[str, Any], token_data: Dict[str, Any]
    ) -> str:
        """
        Create new user session with OAuth tokens.

        Args:
            user_data: User information from YouTube API
            token_data: OAuth token information

        Returns:
            Session ID
        """
        # Generate simple session ID
        session_id = f"session_{datetime.now(timezone.utc).timestamp()}"

        # Calculate token expiry
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # Default 1 hour
        if token_data.get("expiry"):
            try:
                expires_at = datetime.fromisoformat(
                    token_data["expiry"].replace("Z", "+00:00")
                )
            except Exception:
                pass

        session_data = {
            "session_id": session_id,
            "user_id": user_data.get("id", "default_user"),
            "user_info": {
                "youtube_user_id": user_data.get("id"),
                "youtube_name": user_data.get("snippet", {}).get("title"),
                "youtube_email": user_data.get("snippet", {}).get("customUrl"),
                "channel_id": user_data.get("id"),
                "thumbnail_url": user_data.get("snippet", {})
                .get("thumbnails", {})
                .get("default", {})
                .get("url"),
            },
            "tokens": {
                "access_token": self._encrypt_data(token_data.get("access_token", "")),
                "refresh_token": self._encrypt_data(
                    token_data.get("refresh_token", "")
                ),
                "token_uri": token_data.get("token_uri"),
                "client_id": token_data.get("client_id"),
                "client_secret": token_data.get("client_secret"),
                "scopes": token_data.get("scopes", []),
            },
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_active": datetime.now(timezone.utc).isoformat(),
        }

        # Save to file
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        return session_id

    def get_session(self) -> Optional[Dict[str, Any]]:
        """
        Get current user session.

        Returns:
            Session data or None if no valid session
        """
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, "r") as f:
                session_data = json.load(f)

            # Check if session is expired
            expires_at = datetime.fromisoformat(
                session_data["expires_at"].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) > expires_at:
                self.clear_session()
                return None

            # Decrypt tokens
            session_data["tokens"]["access_token"] = self._decrypt_data(
                session_data["tokens"]["access_token"]
            )
            session_data["tokens"]["refresh_token"] = self._decrypt_data(
                session_data["tokens"]["refresh_token"]
            )

            return session_data

        except Exception as e:
            print(f"Error reading session: {e}")
            return None

    def update_session(self, updates: Dict[str, Any]) -> bool:
        """
        Update existing session data.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            True if successful
        """
        session_data = self.get_session()
        if not session_data:
            return False

        # Apply updates
        for key, value in updates.items():
            if key in session_data:
                session_data[key] = value

        # Re-encrypt tokens if they were updated
        if "tokens" in updates:
            if "access_token" in updates["tokens"]:
                session_data["tokens"]["access_token"] = self._encrypt_data(
                    updates["tokens"]["access_token"]
                )
            if "refresh_token" in updates["tokens"]:
                session_data["tokens"]["refresh_token"] = self._encrypt_data(
                    updates["tokens"]["refresh_token"]
                )
        else:
            # Re-encrypt existing tokens
            session_data["tokens"]["access_token"] = self._encrypt_data(
                session_data["tokens"]["access_token"]
            )
            session_data["tokens"]["refresh_token"] = self._encrypt_data(
                session_data["tokens"]["refresh_token"]
            )

        session_data["last_active"] = datetime.now(timezone.utc).isoformat()

        # Save to file
        try:
            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            return False

    def clear_session(self) -> bool:
        """
        Clear current user session.

        Returns:
            True if successful
        """
        try:
            if self.session_file.exists():
                os.remove(self.session_file)
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False

    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated.

        Returns:
            True if valid session exists
        """
        session = self.get_session()
        return session is not None

    def get_user_credentials(self) -> Optional[Dict[str, Any]]:
        """
        Get user's OAuth credentials for API calls.

        Returns:
            Credentials dictionary or None
        """
        session = self.get_session()
        if not session:
            return None

        return {
            "access_token": session["tokens"]["access_token"],
            "refresh_token": session["tokens"]["refresh_token"],
            "token_uri": session["tokens"]["token_uri"],
            "client_id": session["tokens"]["client_id"],
            "client_secret": session["tokens"]["client_secret"],
            "scopes": session["tokens"]["scopes"],
        }

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current user information.

        Returns:
            User info dictionary or None
        """
        session = self.get_session()
        if not session:
            return None

        return session["user_info"]

    def refresh_tokens(self, new_tokens: Dict[str, Any]) -> bool:
        """
        Refresh tokens in the current session.

        Args:
            new_tokens: Dictionary containing new token data

        Returns:
            True if refresh successful, False otherwise
        """
        try:
            session_data = self.get_session()
            if not session_data:
                logger.warning("No active session to refresh tokens")
                return False

            # Update tokens with new values
            if "access_token" in new_tokens:
                session_data["tokens"]["access_token"] = self._encrypt_data(
                    new_tokens["access_token"]
                )

            if "refresh_token" in new_tokens:
                session_data["tokens"]["refresh_token"] = self._encrypt_data(
                    new_tokens["refresh_token"]
                )

            if "expiry" in new_tokens:
                # Update session expiry based on new token expiry
                try:
                    if isinstance(new_tokens["expiry"], str):
                        expires_at = datetime.fromisoformat(
                            new_tokens["expiry"].replace("Z", "+00:00")
                        )
                    else:
                        # Assume it's a datetime object
                        expires_at = new_tokens["expiry"]

                    session_data["expires_at"] = expires_at.isoformat()
                except Exception as expiry_error:
                    logger.warning(f"Failed to parse expiry time: {expiry_error}")
                    # Set default expiry (1 hour from now)
                    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
                    session_data["expires_at"] = expires_at.isoformat()

            # Update last active timestamp
            session_data["last_active"] = datetime.now(timezone.utc).isoformat()

            # Save updated session
            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)

            logger.info("Successfully refreshed session tokens")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh tokens: {e}")
            return False

    def is_token_expired(self) -> bool:
        """
        Check if the current session tokens are expired.

        Returns:
            True if tokens are expired or session doesn't exist
        """
        session_data = self.get_session()
        if not session_data:
            return True

        try:
            expires_at = datetime.fromisoformat(
                session_data["expires_at"].replace("Z", "+00:00")
            )
            return datetime.now(timezone.utc) > expires_at
        except Exception as e:
            logger.error(f"Error checking token expiry: {e}")
            return True


# Global session manager instance
session_manager = SessionManager()

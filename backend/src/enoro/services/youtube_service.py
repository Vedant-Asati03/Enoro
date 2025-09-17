"""
YouTube API service for video discovery and metadata retrieval.
"""

import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import isodate

from ..core.config import settings


class YouTubeService:
    """YouTube API service for video operations."""

    def __init__(self):
        self.api_key = settings.youtube_api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.quota_used = 0
        self.quota_limit = settings.youtube_quota_limit

    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        category_id: Optional[str] = None,
        published_after: Optional[datetime] = None,
        order: str = "relevance",
    ) -> List[str]:
        """
        Search for videos and return video IDs.

        Args:
            query: Search query
            max_results: Maximum number of results
            category_id: YouTube category ID
            published_after: Only videos published after this date
            order: Sort order (relevance, date, rating, viewCount, title)

        Returns:
            List of video IDs
        """
        if not self.api_key:
            raise ValueError("YouTube API key is required")

        params = {
            "key": self.api_key,
            "q": query,
            "part": "id",
            "type": "video",
            "order": order,
            "maxResults": min(max_results, 50),  # YouTube API limit
            "videoDuration": "any",
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
        }

        if category_id:
            params["videoCategoryId"] = category_id

        if published_after:
            params["publishedAfter"] = published_after.isoformat() + "Z"

        try:
            response = requests.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()

            data = response.json()
            self.quota_used += 100  # Search costs 100 quota units

            video_ids = []
            for item in data.get("items", []):
                if "videoId" in item["id"]:
                    video_ids.append(item["id"]["videoId"])

            return video_ids

        except requests.RequestException as e:
            print(f"Error searching videos: {e}")
            return []

    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """
        Get detailed information for a list of video IDs.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            List of video dictionaries with detailed information
        """
        if not video_ids:
            return []

        # YouTube API allows up to 50 IDs per request
        all_videos = []
        chunk_size = 50

        for i in range(0, len(video_ids), chunk_size):
            chunk = video_ids[i : i + chunk_size]
            videos = self._get_video_details_chunk(chunk)
            all_videos.extend(videos)

        return all_videos

    def _get_video_details_chunk(self, video_ids: List[str]) -> List[Dict]:
        """Get video details for a chunk of video IDs."""
        params = {
            "key": self.api_key,
            "id": ",".join(video_ids),
            "part": "snippet,statistics,contentDetails,status",
        }

        try:
            response = requests.get(f"{self.base_url}/videos", params=params)
            response.raise_for_status()

            data = response.json()
            self.quota_used += 1  # Video details costs 1 quota unit per request

            videos = []
            for item in data.get("items", []):
                video = self._parse_video_data(item)
                if self._is_valid_video(video):
                    videos.append(video)

            return videos

        except requests.RequestException as e:
            print(f"Error getting video details: {e}")
            return []

    def _parse_video_data(self, item: Dict) -> Dict:
        """Parse YouTube API video data into our format."""
        snippet = item["snippet"]
        statistics = item.get("statistics", {})
        content_details = item["contentDetails"]
        status = item.get("status", {})

        # Parse duration
        duration_str = content_details.get("duration", "PT0S")
        try:
            duration = isodate.parse_duration(duration_str)
            duration_seconds = int(duration.total_seconds())
        except Exception:
            duration_seconds = 0

        # Parse published date
        try:
            published_at = datetime.fromisoformat(
                snippet["publishedAt"].replace("Z", "+00:00")
            )
        except Exception:
            published_at = datetime.now()

        return {
            "id": item["id"],
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_name": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "thumbnail_url": self._get_best_thumbnail(snippet.get("thumbnails", {})),
            "duration": duration_str,
            "duration_seconds": duration_seconds,
            "published_at": published_at,
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "category_id": int(snippet.get("categoryId", 0)),
            "tags": json.dumps(snippet.get("tags", [])),
            "language": snippet.get(
                "defaultLanguage", snippet.get("defaultAudioLanguage", "en")
            ),
            "url": f"https://www.youtube.com/watch?v={item['id']}",
            "embeddable": status.get("embeddable", True),
            "privacy_status": status.get("privacyStatus", "public"),
        }

    def _get_best_thumbnail(self, thumbnails: Dict) -> str:
        """Get the best available thumbnail URL."""
        if not thumbnails:
            return ""

        # Priority order: maxres, high, medium, default
        for quality in ["maxres", "high", "medium", "default"]:
            if quality in thumbnails:
                return thumbnails[quality]["url"]

        return ""

    def _is_valid_video(self, video: Dict) -> bool:
        """Check if video meets our quality criteria."""
        # Must be public and embeddable
        if video.get("privacy_status") != "public" or not video.get("embeddable", True):
            return False

        # Must have minimum view count (configurable)
        min_views = 1000  # Could be made configurable
        if video.get("view_count", 0) < min_views:
            return False

        # Must have title and description
        if not video.get("title") or len(video.get("title", "")) < 10:
            return False

        # Duration check (avoid very short or very long videos)
        duration = video.get("duration_seconds", 0)
        if duration < 60 or duration > 7200:  # 1 minute to 2 hours
            return False

        return True

    def get_video_categories(self, region_code: str = "US") -> Dict[str, str]:
        """Get YouTube video categories."""
        params = {"key": self.api_key, "part": "snippet", "regionCode": region_code}

        try:
            response = requests.get(f"{self.base_url}/videoCategories", params=params)
            response.raise_for_status()

            data = response.json()
            self.quota_used += 1

            categories = {}
            for item in data.get("items", []):
                categories[item["id"]] = item["snippet"]["title"]

            return categories

        except requests.RequestException as e:
            print(f"Error getting categories: {e}")
            return {}

    def get_trending_videos(
        self, category_id: Optional[str] = None, region_code: str = "US"
    ) -> List[str]:
        """Get trending video IDs."""
        params = {
            "key": self.api_key,
            "part": "id",
            "chart": "mostPopular",
            "regionCode": region_code,
            "maxResults": 50,
        }

        if category_id:
            params["videoCategoryId"] = category_id

        try:
            response = requests.get(f"{self.base_url}/videos", params=params)
            response.raise_for_status()

            data = response.json()
            self.quota_used += 1

            video_ids = []
            for item in data.get("items", []):
                video_ids.append(item["id"])

            return video_ids

        except requests.RequestException as e:
            print(f"Error getting trending videos: {e}")
            return []

    def get_quota_usage(self) -> Dict[str, int]:
        """Get current quota usage."""
        return {
            "used": self.quota_used,
            "limit": self.quota_limit,
            "remaining": max(0, self.quota_limit - self.quota_used),
            "percentage": min(100, int((self.quota_used / self.quota_limit) * 100)),
        }


# Predefined search queries for different categories
SEARCH_QUERIES = {
    "programming": [
        "python tutorial",
        "javascript course",
        "web development",
        "react tutorial",
        "machine learning",
        "data science",
        "programming fundamentals",
        "coding interview",
        "software development",
        "algorithm explanation",
    ],
    "science": [
        "physics explained",
        "chemistry tutorial",
        "biology basics",
        "mathematics lesson",
        "science experiment",
        "space exploration",
        "quantum physics",
        "scientific method",
    ],
    "technology": [
        "tech review",
        "gadget unboxing",
        "smartphone comparison",
        "technology news",
        "innovation showcase",
        "tech tutorial",
        "artificial intelligence",
        "future technology",
    ],
    "education": [
        "educational video",
        "learning techniques",
        "study tips",
        "online course",
        "knowledge sharing",
        "skill development",
        "academic research",
        "educational content",
    ],
    "entertainment": [
        "funny videos",
        "comedy sketches",
        "viral videos",
        "memes compilation",
        "entertainment news",
        "celebrity interviews",
        "movie reviews",
        "gaming videos",
    ],
}


def get_search_queries(category: str) -> List[str]:
    """Get search queries for a specific category."""
    return SEARCH_QUERIES.get(category.lower(), SEARCH_QUERIES["programming"])

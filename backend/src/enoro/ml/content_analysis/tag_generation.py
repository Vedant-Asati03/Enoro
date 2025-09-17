"""
Intelligent tag generation engine for Enoro.

Combines content analysis, user behavior, and collaborative filtering
to generate meaningful tags for channels and user interests.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import Session

from backend.src.enoro.database.models.channel import Channel, UserSubscription
from backend.src.enoro.database.models.tags import (
    ChannelCluster,
    ChannelTag,
    ContentTag,
    TopicCluster,
    UserInterest,
)
from backend.src.enoro.ml.content_analysis.feature_extraction import FeatureExtractor
from backend.src.enoro.ml.content_analysis.topic_modeling import TopicModeler


@dataclass
class TagSuggestion:
    """A suggested tag with confidence and reasoning."""

    name: str
    category: str
    confidence: float
    source: str  # 'content', 'collaborative', 'topic_modeling'
    reasoning: str


@dataclass
class ChannelTagging:
    """Complete tagging results for a channel."""

    channel_id: str
    suggested_tags: List[TagSuggestion]
    primary_category: str
    confidence_score: float


class TagGenerator:
    """
    Intelligent tag generation engine.

    This service combines multiple approaches:
    1. Content-based analysis (descriptions, titles)
    2. Collaborative filtering (user subscription patterns)
    3. Topic modeling (automated theme discovery)
    4. Behavioral analysis (viewing patterns)
    """

    def __init__(self):
        """Initialize the tag generator."""
        self.feature_extractor = FeatureExtractor()
        self.topic_modeler = TopicModeler()

        # Comprehensive predefined tag hierarchies
        self.tag_categories = {
            "content_type": {
                # Educational content
                "tutorial",
                "how_to",
                "guide",
                "walkthrough",
                "course",
                "lesson",
                "masterclass",
                "explanation",
                "demonstration",
                "workshop",
                "training",
                "certification",
                # Review and analysis
                "review",
                "comparison",
                "unboxing",
                "first_impressions",
                "deep_dive",
                "analysis",
                "breakdown",
                "critique",
                "evaluation",
                "recommendation",
                # Entertainment formats
                "vlog",
                "challenge",
                "reaction",
                "compilation",
                "highlight",
                "best_of",
                "funny_moments",
                "fails",
                "pranks",
                "comedy_sketch",
                "parody",
                # News and information
                "news",
                "update",
                "announcement",
                "discussion",
                "debate",
                "interview",
                "documentary",
                "investigative",
                "expose",
                "fact_check",
                # Interactive content
                "live_stream",
                "q_and_a",
                "ama",
                "podcast",
                "webinar",
                "conference",
                "behind_the_scenes",
                "day_in_life",
                "routine",
                # Creative content
                "timelapse",
                "montage",
                "music_video",
                "short_film",
                "animation",
                "art_process",
                "performance",
                "cover",
                "remix",
                "original_content",
                # Informational
                "tips",
                "hacks",
                "secrets",
                "mistakes",
                "myths",
                "facts",
                "history",
                "timeline",
                "evolution",
                "future_predictions",
            },
            "subject_area": {
                # Technology & Programming
                "technology",
                "programming",
                "coding",
                "software_development",
                "web_development",
                "mobile_development",
                "ai",
                "machine_learning",
                "data_science",
                "cybersecurity",
                "blockchain",
                "cryptocurrency",
                "hardware",
                "gadgets",
                "smartphones",
                "computers",
                "robotics",
                "iot",
                "cloud_computing",
                "devops",
                "databases",
                "networking",
                # Gaming
                "gaming",
                "video_games",
                "esports",
                "game_reviews",
                "gameplay",
                "speedrun",
                "game_development",
                "indie_games",
                "retro_gaming",
                "mobile_gaming",
                "streaming",
                "let_s_play",
                "game_theory",
                "gaming_news",
                # Education & Learning
                "education",
                "academic",
                "mathematics",
                "physics",
                "chemistry",
                "biology",
                "history",
                "geography",
                "literature",
                "philosophy",
                "psychology",
                "language_learning",
                "study_tips",
                "exam_prep",
                "school",
                "university",
                # Science & Research
                "science",
                "research",
                "experiments",
                "discoveries",
                "space",
                "astronomy",
                "medicine",
                "health",
                "environmental",
                "climate",
                "nature",
                "wildlife",
                "paleontology",
                "archaeology",
                "anthropology",
                # Arts & Creative
                "art",
                "drawing",
                "painting",
                "digital_art",
                "graphic_design",
                "photography",
                "videography",
                "film_making",
                "animation",
                "3d_modeling",
                "sculpture",
                "crafts",
                "diy",
                "woodworking",
                "pottery",
                "jewelry_making",
                # Music & Audio
                "music",
                "singing",
                "instruments",
                "music_production",
                "recording",
                "music_theory",
                "composition",
                "songwriting",
                "sound_design",
                "classical",
                "rock",
                "pop",
                "hip_hop",
                "electronic",
                "folk",
                "jazz",
                # Business & Finance
                "business",
                "entrepreneurship",
                "startup",
                "marketing",
                "sales",
                "finance",
                "investing",
                "stocks",
                "real_estate",
                "economics",
                "personal_finance",
                "budgeting",
                "taxes",
                "insurance",
                "retirement",
                # Health & Fitness
                "health",
                "fitness",
                "workout",
                "nutrition",
                "diet",
                "weight_loss",
                "muscle_building",
                "cardio",
                "yoga",
                "meditation",
                "mental_health",
                "wellness",
                "medical",
                "sports_medicine",
                "rehabilitation",
                # Lifestyle & Personal
                "lifestyle",
                "fashion",
                "beauty",
                "skincare",
                "makeup",
                "hair",
                "relationships",
                "dating",
                "parenting",
                "family",
                "home",
                "organization",
                "productivity",
                "self_improvement",
                "motivation",
                "spirituality",
                # Food & Cooking
                "cooking",
                "baking",
                "recipes",
                "food_reviews",
                "restaurants",
                "cuisine",
                "vegan",
                "vegetarian",
                "healthy_eating",
                "meal_prep",
                "wine",
                "cocktails",
                "coffee",
                "street_food",
                "fine_dining",
                # Travel & Culture
                "travel",
                "vacation",
                "adventure",
                "culture",
                "languages",
                "countries",
                "cities",
                "backpacking",
                "luxury_travel",
                "budget_travel",
                "solo_travel",
                "family_travel",
                "food_travel",
                "photography_travel",
                # Sports & Recreation
                "sports",
                "football",
                "basketball",
                "soccer",
                "tennis",
                "golf",
                "swimming",
                "running",
                "cycling",
                "martial_arts",
                "extreme_sports",
                "outdoor_activities",
                "hiking",
                "camping",
                "fishing",
                "hunting",
                # Politics & Society
                "politics",
                "government",
                "elections",
                "policy",
                "social_issues",
                "human_rights",
                "activism",
                "law",
                "justice",
                "current_events",
                # Automotive
                "cars",
                "automotive",
                "motorcycles",
                "racing",
                "car_reviews",
                "auto_repair",
                "modifications",
                "electric_vehicles",
                "classic_cars",
                # Home & Garden
                "home_improvement",
                "interior_design",
                "gardening",
                "landscaping",
                "real_estate",
                "architecture",
                "construction",
                "renovation",
            },
            "audience_level": {
                # Skill levels
                "absolute_beginner",
                "beginner",
                "novice",
                "intermediate",
                "advanced",
                "expert",
                "professional",
                "master",
                "guru",
                # Age groups
                "kids",
                "children",
                "preschool",
                "elementary",
                "teens",
                "teenagers",
                "young_adults",
                "adults",
                "middle_aged",
                "seniors",
                "elderly",
                # Experience levels
                "first_time",
                "newcomer",
                "hobbyist",
                "enthusiast",
                "semi_professional",
                "industry_professional",
                "veteran",
                "specialist",
                # Educational levels
                "high_school",
                "college",
                "university",
                "graduate",
                "postgraduate",
                "phd",
                "academic",
                "researcher",
            },
            "content_style": {
                # Tone and mood
                "informative",
                "educational",
                "entertaining",
                "funny",
                "humorous",
                "serious",
                "professional",
                "casual",
                "relaxed",
                "energetic",
                "calm",
                "inspiring",
                "motivational",
                "uplifting",
                "emotional",
                # Presentation style
                "formal",
                "informal",
                "conversational",
                "lecture_style",
                "storytelling",
                "narrative",
                "documentary_style",
                "interview_style",
                "debate_style",
                # Content approach
                "detailed",
                "comprehensive",
                "quick",
                "concise",
                "in_depth",
                "surface_level",
                "practical",
                "theoretical",
                "hands_on",
                "step_by_step",
                # Creative elements
                "creative",
                "artistic",
                "innovative",
                "unique",
                "experimental",
                "traditional",
                "modern",
                "retro",
                "futuristic",
                "minimalist",
                # Engagement style
                "interactive",
                "engaging",
                "thought_provoking",
                "controversial",
                "provocative",
                "neutral",
                "balanced",
                "opinionated",
                "analytical",
            },
            "format": {
                # Duration formats
                "short_form",
                "medium_form",
                "long_form",
                "micro_content",
                "extended",
                "marathon",
                "snippet",
                "highlight",
                "full_length",
                # Series formats
                "series",
                "season",
                "episode",
                "multi_part",
                "standalone",
                "one_off",
                "anthology",
                "documentary_series",
                "course_series",
                # Production styles
                "live_action",
                "animation",
                "stop_motion",
                "time_lapse",
                "slow_motion",
                "screen_recording",
                "slideshow",
                "whiteboard",
                "talking_head",
                # Technical formats
                "4k",
                "hd",
                "vertical",
                "horizontal",
                "square",
                "panoramic",
                "vr",
                "360_degree",
                "split_screen",
                "picture_in_picture",
                # Content structure
                "scripted",
                "unscripted",
                "improvised",
                "structured",
                "free_form",
                "collaborative",
                "solo",
                "group",
                "panel",
                "roundtable",
            },
            "quality_indicators": {
                # Production quality
                "high_production",
                "professional_quality",
                "amateur",
                "homemade",
                "studio_quality",
                "field_recording",
                "polished",
                "raw",
                # Content depth
                "comprehensive",
                "detailed",
                "surface_level",
                "in_depth",
                "thorough",
                "complete",
                "partial",
                "introductory",
                "advanced_level",
                # Accuracy and reliability
                "well_researched",
                "fact_checked",
                "expert_verified",
                "peer_reviewed",
                "opinion_based",
                "speculative",
                "theoretical",
                "evidence_based",
            },
            "engagement_type": {
                # Viewer interaction
                "educational",
                "entertainment",
                "inspirational",
                "instructional",
                "motivational",
                "relaxing",
                "exciting",
                "challenging",
                "comforting",
                # Call to action
                "actionable",
                "follow_along",
                "practice_along",
                "try_at_home",
                "research_further",
                "discuss",
                "share",
                "subscribe",
            },
        }

        # Confidence thresholds for comprehensive tag system
        self.min_confidence = 0.25  # Lower threshold for nuanced tags
        self.high_confidence = 0.75  # Higher bar for high confidence

    def generate_channel_tags(self, db: Session, channel_id: str) -> ChannelTagging:
        """
        Generate comprehensive tags for a channel.

        Args:
            db: Database session
            channel_id: Channel to generate tags for

        Returns:
            ChannelTagging with suggested tags and metadata
        """
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise ValueError(f"Channel {channel_id} not found")

        # Collect tag suggestions from different sources
        content_tags = self._generate_content_based_tags(channel)
        collaborative_tags = self._generate_collaborative_tags(db, channel_id)
        topic_tags = self._generate_topic_based_tags(db, channel_id)

        # Combine and rank all suggestions
        all_suggestions = content_tags + collaborative_tags + topic_tags
        ranked_tags = self._rank_and_deduplicate_tags(all_suggestions)

        # Determine primary category
        primary_category = self._determine_primary_category(ranked_tags)

        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(ranked_tags)

        return ChannelTagging(
            channel_id=channel_id,
            suggested_tags=ranked_tags[:10],  # Top 10 tags
            primary_category=primary_category,
            confidence_score=confidence_score,
        )

    def _generate_content_based_tags(self, channel: Channel) -> List[TagSuggestion]:
        """
        Generate tags based on channel content analysis.

        Args:
            channel: Channel to analyze

        Returns:
            List of content-based tag suggestions
        """
        suggestions = []

        # Analyze channel content
        analysis = self.feature_extractor.analyze_channel_content(channel)

        # Convert keywords to tags
        for keyword in analysis.keywords[:5]:  # Top 5 keywords
            confidence = min(0.8, analysis.confidence_score + 0.2)

            suggestions.append(
                TagSuggestion(
                    name=keyword.lower(),
                    category=self._categorize_tag(keyword),
                    confidence=confidence,
                    source="content",
                    reasoning=f"Extracted from channel description with {confidence:.1%} confidence",
                )
            )

        # Convert content categories to tags
        for category in analysis.content_categories:
            suggestions.append(
                TagSuggestion(
                    name=category,
                    category="subject_area",
                    confidence=0.7,
                    source="content",
                    reasoning=f"Identified as {category} content",
                )
            )

        # Analyze content style from description
        style_tags = self._analyze_content_style(str(channel.description) or "")
        suggestions.extend(style_tags)

        return suggestions

    def _generate_collaborative_tags(
        self, db: Session, channel_id: str
    ) -> List[TagSuggestion]:
        """
        Generate tags based on collaborative filtering.

        Args:
            db: Database session
            channel_id: Channel to analyze

        Returns:
            List of collaborative filtering tag suggestions
        """
        suggestions = []

        # Find users who subscribe to this channel
        subscribers = (
            db.query(UserSubscription.user_id)
            .filter(UserSubscription.channel_id == channel_id)
            .all()
        )

        if not subscribers:
            return suggestions

        user_ids = [sub.user_id for sub in subscribers]

        # Find other channels these users subscribe to
        similar_channels = (
            db.query(UserSubscription.channel_id)
            .filter(
                UserSubscription.user_id.in_(user_ids),
                UserSubscription.channel_id != channel_id,
            )
            .all()
        )

        # Count channel co-occurrences
        channel_counts = Counter(sub.channel_id for sub in similar_channels)

        # Get tags from most similar channels
        for similar_channel_id, count in channel_counts.most_common(5):
            similar_tags = (
                db.query(ChannelTag)
                .filter(ChannelTag.channel_id == similar_channel_id)
                .filter(ChannelTag.relevance_score > 0.5)
                .all()
            )

            for tag_assoc in similar_tags:
                tag = (
                    db.query(ContentTag)
                    .filter(ContentTag.id == tag_assoc.tag_id)
                    .first()
                )
                if tag:
                    # Calculate collaborative confidence based on co-occurrence
                    collab_confidence = min(0.8, count / len(user_ids) * 2)

                    suggestions.append(
                        TagSuggestion(
                            name=str(tag.name),
                            category=str(tag.category),
                            confidence=collab_confidence,
                            source="collaborative",
                            reasoning=f"Suggested by {count} shared subscribers",
                        )
                    )

        return suggestions

    def _generate_topic_based_tags(
        self, db: Session, channel_id: str
    ) -> List[TagSuggestion]:
        """
        Generate tags based on topic modeling results.

        Args:
            db: Database session
            channel_id: Channel to analyze

        Returns:
            List of topic-based tag suggestions
        """
        suggestions = []

        # Find topic clusters this channel belongs to
        clusters = (
            db.query(ChannelCluster)
            .filter(ChannelCluster.channel_id == channel_id)
            .filter(ChannelCluster.probability > 0.3)
            .all()
        )

        for cluster_assoc in clusters:
            cluster = (
                db.query(TopicCluster)
                .filter(TopicCluster.id == cluster_assoc.cluster_id)
                .first()
            )

            if cluster and getattr(cluster, "keywords", None):
                # Extract keywords from cluster
                try:
                    keywords = (
                        json.loads(cluster.keywords)
                        if isinstance(cluster.keywords, str)
                        else cluster.keywords
                    )

                    for keyword in keywords[:3]:  # Top 3 keywords per cluster
                        suggestions.append(
                            TagSuggestion(
                                name=str(keyword),
                                category="topic",
                                confidence=getattr(cluster_assoc, "probability", 0.5),
                                source="topic_modeling",
                                reasoning=f"Part of '{cluster.name}' topic cluster",
                            )
                        )

                except (json.JSONDecodeError, TypeError):
                    # Fallback if keywords aren't properly formatted
                    suggestions.append(
                        TagSuggestion(
                            name=cluster.name.lower(),
                            category="topic",
                            confidence=getattr(cluster_assoc, "probability", 0.5),
                            source="topic_modeling",
                            reasoning=f"Assigned to '{cluster.name}' cluster",
                        )
                    )

        return suggestions

    def _analyze_content_style(self, description: str) -> List[TagSuggestion]:
        """
        Analyze content style from description text.

        Args:
            description: Channel description text

        Returns:
            List of style-based tag suggestions
        """
        suggestions = []

        if not description:
            return suggestions

        description_lower = description.lower()

        # Enhanced style indicators using comprehensive analysis
        style_indicators = {
            # Content types with enhanced detection
            "tutorial": [
                "tutorial",
                "how to",
                "learn",
                "guide",
                "step by step",
                "walkthrough",
                "course",
                "lesson",
                "masterclass",
                "training",
                "workshop",
            ],
            "review": [
                "review",
                "unboxing",
                "test",
                "comparison",
                "opinion",
                "first impressions",
                "deep dive",
                "analysis",
                "breakdown",
                "critique",
                "evaluation",
            ],
            "entertainment": [
                "fun",
                "funny",
                "entertaining",
                "comedy",
                "laugh",
                "challenge",
                "reaction",
                "compilation",
                "highlight",
                "best of",
                "fails",
            ],
            "educational": [
                "education",
                "teach",
                "explain",
                "knowledge",
                "academic",
                "study",
                "facts",
                "history",
                "science",
                "research",
                "documentary",
            ],
            "vlog": [
                "vlog",
                "daily",
                "life",
                "personal",
                "journey",
                "day in life",
                "routine",
                "behind the scenes",
                "lifestyle",
                "experience",
            ],
            "news": [
                "news",
                "updates",
                "latest",
                "breaking",
                "current",
                "announcement",
                "discussion",
                "interview",
                "investigative",
            ],
            "professional": [
                "professional",
                "business",
                "official",
                "corporate",
                "industry",
                "expert",
                "certification",
                "advanced",
                "technical",
            ],
            # Emotional and engagement styles
            "inspirational": [
                "inspire",
                "motivate",
                "dream",
                "achieve",
                "success",
                "overcome",
                "believe",
                "transform",
                "change your life",
                "potential",
            ],
            "casual": [
                "casual",
                "chill",
                "relaxed",
                "hang out",
                "friends",
                "informal",
                "conversational",
                "laid back",
                "easy going",
            ],
            "energetic": [
                "energy",
                "exciting",
                "pumped",
                "hyped",
                "intense",
                "fast paced",
                "action packed",
                "dynamic",
                "vibrant",
            ],
            "calm": [
                "calm",
                "peaceful",
                "relaxing",
                "meditation",
                "zen",
                "mindful",
                "soothing",
                "tranquil",
                "serene",
            ],
            # Content depth and complexity
            "beginner_friendly": [
                "beginner",
                "start",
                "basics",
                "introduction",
                "first time",
                "easy",
                "simple",
                "for dummies",
                "101",
                "getting started",
            ],
            "advanced": [
                "advanced",
                "expert",
                "professional",
                "complex",
                "deep dive",
                "technical",
                "sophisticated",
                "high level",
                "masterclass",
            ],
            "comprehensive": [
                "complete",
                "everything",
                "full",
                "comprehensive",
                "thorough",
                "detailed",
                "in-depth",
                "extensive",
                "ultimate guide",
            ],
            "quick": [
                "quick",
                "fast",
                "rapid",
                "speed",
                "brief",
                "summary",
                "highlights",
                "key points",
                "short",
                "concise",
            ],
        }

        for style, indicators in style_indicators.items():
            matches = sum(
                1 for indicator in indicators if indicator in description_lower
            )

            if matches > 0:
                confidence = min(0.8, matches / len(indicators) + 0.3)

                suggestions.append(
                    TagSuggestion(
                        name=style,
                        category="content_style",
                        confidence=confidence,
                        source="content",
                        reasoning="Style indicators found in description",
                    )
                )

        return suggestions

    def _categorize_tag(self, tag_name: str) -> str:
        """
        Categorize a tag name into predefined categories.

        Args:
            tag_name: Name of the tag

        Returns:
            Category name
        """
        tag_lower = tag_name.lower()

        for category, tags in self.tag_categories.items():
            if tag_lower in tags:
                return category

        # Check for partial matches in subject areas
        for subject in self.tag_categories["subject_area"]:
            if subject in tag_lower or tag_lower in subject:
                return "subject_area"

        return "general"  # Default category

    def _rank_and_deduplicate_tags(
        self, suggestions: List[TagSuggestion]
    ) -> List[TagSuggestion]:
        """
        Rank tag suggestions and remove duplicates.

        Args:
            suggestions: List of tag suggestions

        Returns:
            Ranked and deduplicated list
        """
        # Group by tag name and combine confidences
        tag_groups = defaultdict(list)
        for suggestion in suggestions:
            tag_groups[suggestion.name].append(suggestion)

        # Combine duplicate tags
        combined_tags = []
        for tag_name, group in tag_groups.items():
            if len(group) == 1:
                combined_tags.append(group[0])
            else:
                # Combine multiple suggestions for the same tag
                best_suggestion = max(group, key=lambda x: x.confidence)

                # Boost confidence if multiple sources agree
                confidence_boost = min(0.2, (len(group) - 1) * 0.1)
                combined_confidence = min(
                    1.0, best_suggestion.confidence + confidence_boost
                )

                # Combine sources and reasoning
                sources = list(set(s.source for s in group))
                reasoning = f"Confirmed by {len(group)} sources: {', '.join(sources)}"

                combined_tags.append(
                    TagSuggestion(
                        name=tag_name,
                        category=best_suggestion.category,
                        confidence=combined_confidence,
                        source="+".join(sources),
                        reasoning=reasoning,
                    )
                )

        # Filter by minimum confidence and sort
        filtered_tags = [
            tag for tag in combined_tags if tag.confidence >= self.min_confidence
        ]
        return sorted(filtered_tags, key=lambda x: x.confidence, reverse=True)

    def _determine_primary_category(self, tags: List[TagSuggestion]) -> str:
        """
        Determine the primary category for a channel.

        Args:
            tags: List of ranked tag suggestions

        Returns:
            Primary category name
        """
        if not tags:
            return "general"

        # Count tags by category, weighted by confidence
        category_scores = defaultdict(float)
        for tag in tags:
            category_scores[tag.category] += tag.confidence

        # Return category with highest score
        return (
            max(category_scores.items(), key=lambda x: x[1])[0]
            if category_scores
            else "general"
        )

    def _calculate_overall_confidence(self, tags: List[TagSuggestion]) -> float:
        """
        Calculate overall confidence in the tagging results.

        Args:
            tags: List of tag suggestions

        Returns:
            Overall confidence score (0-1)
        """
        if not tags:
            return 0.0

        # Weighted average of top tag confidences
        top_tags = tags[:5]  # Top 5 tags
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][: len(top_tags)]

        weighted_sum = sum(
            tag.confidence * weight for tag, weight in zip(top_tags, weights)
        )
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def save_channel_tags(self, db: Session, channel_tagging: ChannelTagging) -> None:
        """
        Save generated tags to the database.

        Args:
            db: Database session
            channel_tagging: Tagging results to save
        """
        # Remove existing tags for this channel
        db.query(ChannelTag).filter(
            ChannelTag.channel_id == channel_tagging.channel_id
        ).delete()

        for suggestion in channel_tagging.suggested_tags:
            # Create or get content tag
            content_tag = (
                db.query(ContentTag).filter(ContentTag.name == suggestion.name).first()
            )

            if not content_tag:
                content_tag = ContentTag(
                    name=suggestion.name,
                    category=suggestion.category,
                    description=f"Auto-generated tag: {suggestion.reasoning}",
                    confidence_score=suggestion.confidence,
                )
                db.add(content_tag)
                db.flush()  # Get the ID

            # Create channel-tag association
            channel_tag = ChannelTag(
                channel_id=channel_tagging.channel_id,
                tag_id=content_tag.id,
                relevance_score=suggestion.confidence,
                confidence_score=suggestion.confidence,
                source=suggestion.source,
                analysis_version="1.0",
            )
            db.add(channel_tag)

        db.commit()

    def generate_user_interest_tags(
        self, db: Session, user_id: str
    ) -> List[TagSuggestion]:
        """
        Generate interest tags for a user based on their subscriptions.

        Args:
            db: Database session
            user_id: User to analyze

        Returns:
            List of user interest tag suggestions
        """
        # Get user subscriptions
        subscriptions = (
            db.query(UserSubscription).filter(UserSubscription.user_id == user_id).all()
        )

        if not subscriptions:
            return []

        # Collect tags from subscribed channels
        interest_scores = defaultdict(float)
        tag_sources = defaultdict(list)

        for subscription in subscriptions:
            channel_tags = (
                db.query(ChannelTag)
                .filter(ChannelTag.channel_id == subscription.channel_id)
                .filter(ChannelTag.relevance_score > 0.4)
                .all()
            )

            for channel_tag in channel_tags:
                tag = (
                    db.query(ContentTag)
                    .filter(ContentTag.id == channel_tag.tag_id)
                    .first()
                )
                if tag:
                    # Use getattr to access the attribute safely
                    score = getattr(channel_tag, "relevance_score", 0.0)
                    interest_scores[tag.name] += score
                    tag_sources[tag.name].append(subscription.channel_id)

        # Convert to tag suggestions
        suggestions = []
        for tag_name, score in interest_scores.items():
            # Normalize score by number of subscriptions
            normalized_score = min(1.0, score / len(subscriptions))

            if normalized_score >= self.min_confidence:
                tag = db.query(ContentTag).filter(ContentTag.name == tag_name).first()

                suggestions.append(
                    TagSuggestion(
                        name=tag_name,
                        category=str(tag.category) if tag else "general",
                        confidence=normalized_score,
                        source="subscriptions",
                        reasoning=f"Interest inferred from {len(tag_sources[tag_name])} subscribed channels",
                    )
                )

        # Sort by confidence
        return sorted(suggestions, key=lambda x: x.confidence, reverse=True)

    def save_user_interests(
        self, db: Session, user_id: str, interest_tags: List[TagSuggestion]
    ) -> None:
        """
        Save user interest tags to the database.

        Args:
            db: Database session
            user_id: User ID
            interest_tags: List of interest tag suggestions
        """
        # Remove existing interests for this user
        db.query(UserInterest).filter(UserInterest.user_id == user_id).delete()

        for suggestion in interest_tags:
            # Get content tag
            content_tag = (
                db.query(ContentTag).filter(ContentTag.name == suggestion.name).first()
            )

            if content_tag:
                # Create user interest
                user_interest = UserInterest(
                    user_id=user_id,
                    tag_id=content_tag.id,
                    interest_score=suggestion.confidence,
                    engagement_score=suggestion.confidence,  # Simplified
                    calculated_from={
                        "source": suggestion.source,
                        "reasoning": suggestion.reasoning,
                    },
                )
                db.add(user_interest)

        db.commit()

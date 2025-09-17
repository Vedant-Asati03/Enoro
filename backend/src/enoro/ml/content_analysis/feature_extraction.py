"""
Feature extraction service for content analysis and tag generation.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import nltk
import numpy as np
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from backend.src.enoro.database.models.channel import Channel


@dataclass
class TextFeatures:
    """Container for extracted text features."""

    # Basic text metrics
    word_count: int
    sentence_count: int
    avg_word_length: float
    readability_score: float

    # Content features
    keywords: List[str]
    entities: List[str]
    topics: List[str]
    sentiment_score: float

    # Technical features
    tfidf_vector: np.ndarray
    topic_distribution: np.ndarray


@dataclass
class ContentAnalysis:
    """Complete content analysis results."""

    channel_id: str
    primary_topics: List[str]
    content_categories: List[str]
    keywords: List[str]
    description_features: TextFeatures
    title_features: Optional[TextFeatures]
    confidence_score: float


class FeatureExtractor:
    """
    Extract features from YouTube content for ML analysis.

    This service processes channel descriptions, video titles, and other text
    content to generate features for topic modeling and tag generation.
    """

    def __init__(self):
        """Initialize the feature extractor with NLP tools."""
        self._download_nltk_data()

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Add domain-specific stop words
        self.stop_words.update(
            {
                "youtube",
                "channel",
                "video",
                "subscribe",
                "like",
                "comment",
                "share",
                "watch",
                "follow",
                "content",
                "creator",
                "uploading",
            }
        )

        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )

        self.count_vectorizer = CountVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2)
        )

        # Topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=20, random_state=42, learning_method="online", max_iter=10
        )

        # Comprehensive content categories with extensive keyword coverage
        self.content_categories = {
            "technology": [
                # General tech
                "tech",
                "technology",
                "software",
                "hardware",
                "digital",
                "innovation",
                "gadgets",
                "devices",
                "electronics",
                "computer",
                "pc",
                "laptop",
                # Programming & Development
                "programming",
                "coding",
                "development",
                "developer",
                "code",
                "script",
                "algorithm",
                "software development",
                "web development",
                "mobile development",
                "frontend",
                "backend",
                "fullstack",
                "devops",
                "api",
                "database",
                "framework",
                "library",
                "programming language",
                "javascript",
                "python",
                "java",
                "react",
                "node",
                "html",
                "css",
                "sql",
                "git",
                "github",
                # AI & ML
                "ai",
                "artificial intelligence",
                "machine learning",
                "ml",
                "deep learning",
                "neural network",
                "data science",
                "analytics",
                "big data",
                "automation",
                "chatbot",
                "computer vision",
                "nlp",
                "natural language processing",
                # Emerging Tech
                "blockchain",
                "cryptocurrency",
                "bitcoin",
                "ethereum",
                "nft",
                "web3",
                "metaverse",
                "vr",
                "virtual reality",
                "ar",
                "augmented reality",
                "iot",
                "internet of things",
                "5g",
                "cloud computing",
                "cybersecurity",
                # Consumer Tech
                "smartphone",
                "iphone",
                "android",
                "tablet",
                "smartwatch",
                "headphones",
                "camera",
                "gaming console",
                "smart home",
                "alexa",
                "google assistant",
            ],
            "gaming": [
                # General Gaming
                "game",
                "gaming",
                "gameplay",
                "gamer",
                "video game",
                "video games",
                "play",
                "player",
                "gaming community",
                "gaming culture",
                # Gaming Content
                "stream",
                "streaming",
                "twitch",
                "let's play",
                "walkthrough",
                "playthrough",
                "speedrun",
                "tutorial",
                "guide",
                "tips",
                "tricks",
                "strategy",
                # Gaming Platforms
                "pc gaming",
                "console",
                "playstation",
                "xbox",
                "nintendo",
                "switch",
                "steam",
                "epic games",
                "mobile gaming",
                "browser game",
                # Gaming Genres
                "fps",
                "mmo",
                "rpg",
                "rts",
                "puzzle",
                "platformer",
                "racing",
                "sports game",
                "fighting game",
                "battle royale",
                "moba",
                "indie game",
                "aaa game",
                # Esports & Competitive
                "esports",
                "competitive",
                "tournament",
                "championship",
                "pro gaming",
                "league",
                "team",
                "clan",
                "guild",
                "ranking",
                "leaderboard",
                # Gaming Hardware
                "gaming pc",
                "graphics card",
                "processor",
                "gaming chair",
                "gaming setup",
                "mechanical keyboard",
                "gaming mouse",
                "headset",
                "monitor",
            ],
            "education": [
                # General Education
                "tutorial",
                "learn",
                "learning",
                "education",
                "educational",
                "course",
                "lesson",
                "class",
                "teach",
                "teaching",
                "instructor",
                "professor",
                "study",
                "studying",
                "student",
                "school",
                "university",
                "college",
                # Educational Content
                "how to",
                "guide",
                "explanation",
                "demonstration",
                "workshop",
                "seminar",
                "lecture",
                "masterclass",
                "certification",
                "training",
                "skill",
                "knowledge",
                "information",
                "facts",
                "research",
                "academic",
                # Subjects
                "math",
                "mathematics",
                "physics",
                "chemistry",
                "biology",
                "science",
                "history",
                "geography",
                "literature",
                "language",
                "english",
                "spanish",
                "french",
                "art",
                "music",
                "philosophy",
                "psychology",
                # Learning Methods
                "online learning",
                "e-learning",
                "distance learning",
                "homeschool",
                "exam",
                "test",
                "quiz",
                "homework",
                "assignment",
                "project",
                "study tips",
                "exam prep",
                "notes",
                "revision",
            ],
            "entertainment": [
                # Comedy & Humor
                "funny",
                "comedy",
                "humor",
                "hilarious",
                "laugh",
                "laughter",
                "joke",
                "jokes",
                "meme",
                "memes",
                "parody",
                "satire",
                "sketch",
                "stand-up",
                "comedian",
                "comic",
                "amusing",
                "entertaining",
                # General Entertainment
                "entertainment",
                "fun",
                "enjoy",
                "enjoyment",
                "leisure",
                "hobby",
                "recreation",
                "activity",
                "event",
                "show",
                "performance",
                # Content Types
                "reaction",
                "compilation",
                "highlight",
                "best of",
                "fails",
                "fail",
                "prank",
                "pranks",
                "challenge",
                "viral",
                "trending",
                "popular",
                # Interactive Entertainment
                "game show",
                "quiz",
                "contest",
                "competition",
                "interactive",
                "audience",
                "participation",
                "live",
                "stream",
            ],
            "music": [
                # General Music
                "music",
                "musical",
                "song",
                "songs",
                "track",
                "album",
                "single",
                "artist",
                "musician",
                "singer",
                "band",
                "group",
                "composer",
                # Music Genres
                "rock",
                "pop",
                "hip hop",
                "rap",
                "jazz",
                "classical",
                "electronic",
                "country",
                "folk",
                "blues",
                "reggae",
                "metal",
                "punk",
                "indie",
                "alternative",
                "r&b",
                "soul",
                "funk",
                "disco",
                "house",
                "techno",
                # Music Production
                "music production",
                "recording",
                "studio",
                "mixing",
                "mastering",
                "beat",
                "melody",
                "harmony",
                "rhythm",
                "composition",
                "songwriting",
                # Instruments & Performance
                "instrument",
                "instruments",
                "guitar",
                "piano",
                "drums",
                "violin",
                "bass",
                "keyboard",
                "microphone",
                "concert",
                "live music",
                "performance",
                "cover",
                "acoustic",
                "electric",
                "orchestra",
                "symphony",
            ],
            "lifestyle": [
                # Daily Life
                "vlog",
                "vlogging",
                "lifestyle",
                "daily",
                "routine",
                "life",
                "personal",
                "day in the life",
                "morning routine",
                "evening routine",
                "habits",
                # Personal Development
                "self improvement",
                "motivation",
                "inspiration",
                "productivity",
                "organization",
                "mindfulness",
                "meditation",
                "wellness",
                "balance",
                # Relationships & Social
                "relationships",
                "dating",
                "friendship",
                "family",
                "parenting",
                "social",
                "community",
                "culture",
                "tradition",
                "celebration",
                # Home & Living
                "home",
                "apartment",
                "house",
                "interior design",
                "decoration",
                "furniture",
                "cleaning",
                "organization",
                "diy",
                "home improvement",
            ],
            "business": [
                # General Business
                "business",
                "entrepreneur",
                "entrepreneurship",
                "startup",
                "company",
                "corporate",
                "professional",
                "career",
                "work",
                "job",
                "employment",
                # Finance & Investment
                "finance",
                "money",
                "investment",
                "investing",
                "stocks",
                "trading",
                "cryptocurrency",
                "bitcoin",
                "personal finance",
                "budgeting",
                "savings",
                "retirement",
                "insurance",
                "taxes",
                "real estate",
                # Marketing & Sales
                "marketing",
                "advertising",
                "promotion",
                "branding",
                "sales",
                "customer",
                "client",
                "social media marketing",
                "digital marketing",
                "seo",
                "content marketing",
                "email marketing",
                # Business Operations
                "management",
                "leadership",
                "strategy",
                "planning",
                "analytics",
                "productivity",
                "efficiency",
                "automation",
                "operations",
                "logistics",
            ],
            "health": [
                # Fitness & Exercise
                "fitness",
                "workout",
                "exercise",
                "training",
                "gym",
                "cardio",
                "strength training",
                "weightlifting",
                "yoga",
                "pilates",
                "running",
                "cycling",
                "swimming",
                "sport",
                "sports",
                "athletic",
                "athlete",
                # Nutrition & Diet
                "nutrition",
                "diet",
                "healthy eating",
                "meal prep",
                "protein",
                "vitamins",
                "supplements",
                "weight loss",
                "weight gain",
                "calories",
                "organic",
                "vegan",
                "vegetarian",
                "keto",
                "paleo",
                # Health & Wellness
                "health",
                "wellness",
                "medical",
                "healthcare",
                "doctor",
                "medicine",
                "mental health",
                "therapy",
                "counseling",
                "stress",
                "anxiety",
                "depression",
                "self care",
                "recovery",
                "rehabilitation",
                # Beauty & Skincare
                "beauty",
                "skincare",
                "makeup",
                "cosmetics",
                "hair",
                "haircare",
                "salon",
                "spa",
                "facial",
                "massage",
                "grooming",
            ],
            "travel": [
                # General Travel
                "travel",
                "traveling",
                "trip",
                "vacation",
                "holiday",
                "journey",
                "explore",
                "exploration",
                "adventure",
                "destination",
                "tourism",
                "tourist",
                "backpacking",
                "road trip",
                "cruise",
                "flight",
                # Travel Types
                "solo travel",
                "family travel",
                "budget travel",
                "luxury travel",
                "business travel",
                "group travel",
                "international travel",
                "domestic travel",
                "weekend getaway",
                "honeymoon",
                # Destinations & Geography
                "city",
                "country",
                "continent",
                "beach",
                "mountain",
                "desert",
                "forest",
                "island",
                "lake",
                "river",
                "national park",
                "landmark",
                "monument",
                "museum",
                "hotel",
                "resort",
                "hostel",
                # Travel Activities
                "sightseeing",
                "photography",
                "hiking",
                "camping",
                "safari",
                "diving",
                "snorkeling",
                "skiing",
                "surfing",
                "culture",
                "history",
                "food tour",
                "wine tasting",
                "shopping",
                "nightlife",
            ],
            "cooking": [
                # General Cooking
                "cooking",
                "cook",
                "recipe",
                "recipes",
                "food",
                "kitchen",
                "chef",
                "culinary",
                "baking",
                "baker",
                "preparation",
                "meal",
                "dish",
                # Cooking Techniques
                "grilling",
                "roasting",
                "frying",
                "steaming",
                "boiling",
                "sauteing",
                "braising",
                "smoking",
                "fermentation",
                "marinating",
                "seasoning",
                # Cuisine Types
                "italian",
                "chinese",
                "mexican",
                "indian",
                "french",
                "japanese",
                "thai",
                "mediterranean",
                "american",
                "korean",
                "vietnamese",
                "fusion",
                "traditional",
                "modern",
                "street food",
                "fine dining",
                # Dietary Preferences
                "vegan",
                "vegetarian",
                "gluten free",
                "dairy free",
                "keto",
                "paleo",
                "low carb",
                "healthy",
                "organic",
                "farm to table",
                "sustainable",
                # Food Categories
                "appetizer",
                "main course",
                "dessert",
                "breakfast",
                "lunch",
                "dinner",
                "snack",
                "beverage",
                "cocktail",
                "wine",
                "coffee",
                "tea",
                "bread",
                "pasta",
                "pizza",
                "salad",
                "soup",
                "sauce",
                "spice",
                "herb",
            ],
            "science": [
                # General Science
                "science",
                "scientific",
                "research",
                "experiment",
                "laboratory",
                "discovery",
                "innovation",
                "breakthrough",
                "theory",
                "hypothesis",
                "data",
                "analysis",
                "methodology",
                "peer review",
                # Physical Sciences
                "physics",
                "chemistry",
                "astronomy",
                "astrophysics",
                "cosmology",
                "quantum",
                "relativity",
                "particle physics",
                "thermodynamics",
                "electromagnetism",
                "optics",
                "materials science",
                # Life Sciences
                "biology",
                "genetics",
                "evolution",
                "ecology",
                "botany",
                "zoology",
                "microbiology",
                "biochemistry",
                "molecular biology",
                "cell biology",
                "neuroscience",
                "psychology",
                "cognitive science",
                # Earth & Environmental Sciences
                "geology",
                "meteorology",
                "climatology",
                "oceanography",
                "environmental science",
                "climate change",
                "sustainability",
                "renewable energy",
                "conservation",
                "biodiversity",
                # Applied Sciences
                "engineering",
                "technology",
                "medical science",
                "pharmaceutical",
                "biotechnology",
                "nanotechnology",
                "robotics",
                "space science",
            ],
            "art": [
                # Visual Arts
                "art",
                "artistic",
                "artist",
                "drawing",
                "painting",
                "sketch",
                "illustration",
                "design",
                "graphic design",
                "digital art",
                "sculpture",
                "photography",
                "photo",
                "picture",
                "gallery",
                "exhibition",
                "museum",
                "canvas",
                "brush",
                "paint",
                "color",
                # Creative Arts
                "creative",
                "creativity",
                "imagination",
                "inspiration",
                "expression",
                "aesthetic",
                "beauty",
                "style",
                "technique",
                "masterpiece",
                # Art Forms
                "abstract",
                "realism",
                "impressionism",
                "modern art",
                "contemporary",
                "fine art",
                "street art",
                "graffiti",
                "mural",
                "portrait",
                "landscape",
                "still life",
                "animation",
                "cartoon",
                "comic",
                # Crafts & DIY
                "craft",
                "crafting",
                "handmade",
                "diy",
                "pottery",
                "ceramics",
                "jewelry",
                "woodworking",
                "sewing",
                "knitting",
                "embroidery",
                "origami",
                "calligraphy",
                "printmaking",
            ],
            "sports": [
                # General Sports
                "sport",
                "sports",
                "athletic",
                "athlete",
                "team",
                "competition",
                "championship",
                "tournament",
                "league",
                "season",
                "game",
                "match",
                "training",
                "practice",
                "coach",
                "coaching",
                "fitness",
                "performance",
                # Popular Sports
                "football",
                "soccer",
                "basketball",
                "baseball",
                "tennis",
                "golf",
                "swimming",
                "running",
                "cycling",
                "boxing",
                "wrestling",
                "martial arts",
                "hockey",
                "volleyball",
                "badminton",
                "table tennis",
                "cricket",
                # Extreme & Adventure Sports
                "extreme sports",
                "skateboarding",
                "snowboarding",
                "skiing",
                "surfing",
                "rock climbing",
                "mountain biking",
                "parkour",
                "skydiving",
                "bungee",
                # Olympic & International
                "olympics",
                "world cup",
                "international",
                "professional",
                "amateur",
                "college sports",
                "high school sports",
                "youth sports",
                "paralympics",
            ],
            "news": [
                # General News
                "news",
                "breaking news",
                "current events",
                "journalism",
                "reporter",
                "journalist",
                "media",
                "press",
                "broadcast",
                "update",
                "headlines",
                # News Categories
                "politics",
                "political",
                "government",
                "election",
                "policy",
                "law",
                "economy",
                "economic",
                "finance",
                "market",
                "business news",
                "international",
                "world news",
                "local news",
                "weather",
                "sports news",
                # Social Issues
                "social issues",
                "human rights",
                "equality",
                "justice",
                "activism",
                "protest",
                "movement",
                "society",
                "community",
                "public health",
                "environment",
                "climate",
                "sustainability",
            ],
        }

        self._is_fitted = False

    def _download_nltk_data(self):
        """Download required NLTK data if not already present."""
        try:
            required_data = [
                "punkt",
                "stopwords",
                "wordnet",
                "averaged_perceptron_tagger",
                "maxent_ne_chunker",
                "words",
                "vader_lexicon",
            ]

            for data in required_data:
                try:
                    nltk.data.find(f"tokenizers/{data}")
                except LookupError:
                    nltk.download(data, quiet=True)

        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove special characters but keep spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text using TF-IDF.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        if not text or len(text.strip()) < 10:
            return []

        try:
            # Tokenize and clean
            words = word_tokenize(self.preprocess_text(text))
            words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words and len(word) > 2
            ]

            # Get word frequencies
            word_freq = Counter(words)

            # Return most common words
            return [word for word, _ in word_freq.most_common(max_keywords)]

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of named entities
        """
        if not text:
            return []

        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Extract named entities
            chunks = ne_chunk(pos_tags)
            entities = []

            for chunk in chunks:
                if hasattr(chunk, "label"):
                    entity = " ".join([token for token, pos in chunk.leaves()])
                    entities.append(entity)

            return list(set(entities))

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def classify_content_category(self, text: str) -> List[str]:
        """
        Classify content into predefined categories.

        Args:
            text: Content text to classify

        Returns:
            List of matching content categories
        """
        if not text:
            return []

        text_lower = text.lower()
        matched_categories = []

        for category, keywords in self.content_categories.items():
            # Check if any category keywords appear in text
            if any(keyword in text_lower for keyword in keywords):
                matched_categories.append(category)

        return matched_categories

    def calculate_readability(self, text: str) -> float:
        """
        Calculate text readability score (simplified Flesch score).

        Args:
            text: Input text

        Returns:
            Readability score (0-100, higher = easier to read)
        """
        if not text or len(text.strip()) < 10:
            return 50.0  # Default middle score

        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)

            if len(sentences) == 0 or len(words) == 0:
                return 50.0

            # Calculate average sentence length and syllable count
            avg_sentence_length = len(words) / len(sentences)

            # Simplified syllable count (vowel groups)
            syllable_count = sum(
                len(re.findall(r"[aeiouAEIOU]+", word)) for word in words
            )
            avg_syllables_per_word = (
                syllable_count / len(words) if len(words) > 0 else 1
            )

            # Simplified Flesch Reading Ease Score
            score = (
                206.835
                - (1.015 * avg_sentence_length)
                - (84.6 * avg_syllables_per_word)
            )

            # Clamp between 0 and 100
            return max(0, min(100, score))

        except Exception as e:
            print(f"Error calculating readability: {e}")
            return 50.0

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using VADER sentiment analyzer.

        Args:
            text: Input text

        Returns:
            Sentiment score (-1 to 1, negative to positive)
        """
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)

            # Return compound score (-1 to 1)
            return scores["compound"]

        except Exception as e:
            print(f"Error calculating sentiment: {e}")
            return 0.0  # Neutral sentiment as fallback

    def extract_text_features(self, text: str) -> TextFeatures:
        """
        Extract comprehensive features from text.

        Args:
            text: Input text to analyze

        Returns:
            TextFeatures object with extracted features
        """
        if not text:
            return TextFeatures(
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                readability_score=50.0,
                keywords=[],
                entities=[],
                topics=[],
                sentiment_score=0.0,
                tfidf_vector=np.array([]),
                topic_distribution=np.array([]),
            )

        # Basic metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0.0

        # Extract features
        keywords = self.extract_keywords(text)
        entities = self.extract_entities(text)
        topics = self.classify_content_category(text)
        readability_score = self.calculate_readability(text)
        sentiment_score = self.calculate_sentiment(text)

        # Vectorization (empty for now, will be filled when fitting models)
        tfidf_vector = np.array([])
        topic_distribution = np.array([])

        return TextFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            readability_score=readability_score,
            keywords=keywords,
            entities=entities,
            topics=topics,
            sentiment_score=sentiment_score,
            tfidf_vector=tfidf_vector,
            topic_distribution=topic_distribution,
        )

    def analyze_channel_content(self, channel: Channel) -> ContentAnalysis:
        """
        Analyze a channel's content and extract comprehensive features.

        Args:
            channel: Channel database model

        Returns:
            ContentAnalysis with extracted features and topics
        """
        # Extract features from channel description
        description_features = self.extract_text_features(channel.description or "")

        # Combine keywords and topics
        all_keywords = description_features.keywords
        all_topics = description_features.topics

        # Calculate confidence score based on available data
        confidence_score = self._calculate_confidence_score(
            channel, description_features
        )

        return ContentAnalysis(
            channel_id=channel.id,
            primary_topics=all_topics[:5],  # Top 5 topics
            content_categories=all_topics,
            keywords=all_keywords,
            description_features=description_features,
            title_features=None,  # Could add channel name analysis
            confidence_score=confidence_score,
        )

    def _calculate_confidence_score(
        self, channel: Channel, features: TextFeatures
    ) -> float:
        """
        Calculate confidence score for the analysis.

        Args:
            channel: Channel model
            features: Extracted text features

        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.0

        # Base score from text length
        if features.word_count > 20:
            score += 0.4
        elif features.word_count > 10:
            score += 0.2

        # Keywords found
        if len(features.keywords) > 3:
            score += 0.3
        elif len(features.keywords) > 0:
            score += 0.1

        # Topics identified
        if len(features.topics) > 0:
            score += 0.3

        # Channel metadata available
        if channel.subscriber_count > 1000:
            score += 0.1

        return min(1.0, score)

    def fit_models(self, channels: List[Channel]) -> None:
        """
        Fit TF-IDF and topic models on channel data.

        Args:
            channels: List of channels to train on
        """
        # Collect all text data
        texts = []
        for channel in channels:
            if channel.description and len(channel.description.strip()) > 10:
                texts.append(self.preprocess_text(channel.description))

        if len(texts) < 5:  # Need minimum data to train
            print("Warning: Insufficient text data to train models")
            return

        try:
            # Fit TF-IDF vectorizer
            self.tfidf_vectorizer.fit(texts)

            # Fit topic model
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            self.lda_model.fit(tfidf_matrix)

            self._is_fitted = True
            print(f"Successfully fitted models on {len(texts)} channel descriptions")

        except Exception as e:
            print(f"Error fitting models: {e}")

    def generate_channel_vector(self, channel: Channel) -> Optional[np.ndarray]:
        """
        Generate TF-IDF vector for a channel.

        Args:
            channel: Channel to vectorize

        Returns:
            TF-IDF vector or None if models not fitted
        """
        if not self._is_fitted or not channel.description:
            return None

        try:
            processed_text = self.preprocess_text(channel.description)
            return self.tfidf_vectorizer.transform([processed_text]).toarray()[0]
        except Exception as e:
            print(f"Error generating vector for channel {channel.id}: {e}")
            return None

    def get_topic_distribution(self, channel: Channel) -> Optional[np.ndarray]:
        """
        Get topic distribution for a channel.

        Args:
            channel: Channel to analyze

        Returns:
            Topic probability distribution or None if models not fitted
        """
        if not self._is_fitted or not channel.description:
            return None

        try:
            processed_text = self.preprocess_text(channel.description)
            tfidf_vector = self.tfidf_vectorizer.transform([processed_text])
            return self.lda_model.transform(tfidf_vector)[0]
        except Exception as e:
            print(f"Error getting topic distribution for channel {channel.id}: {e}")
            return None

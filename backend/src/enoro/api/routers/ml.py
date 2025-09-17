"""
FastAPI routes for ML-powered content analysis and recommendations.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.src.enoro.database.models import get_db
from backend.src.enoro.ml import (
    TagGenerator,
    TopicModeler,
    FeatureExtractor,
    RecommendationEngine,
)


# Pydantic models for API responses
class TagSuggestionResponse(BaseModel):
    """API response model for tag suggestions."""

    name: str
    category: str
    confidence: float
    source: str
    reasoning: str


class ChannelTaggingResponse(BaseModel):
    """API response model for channel tagging results."""

    channel_id: str
    suggested_tags: List[TagSuggestionResponse]
    primary_category: str
    confidence_score: float


class UserInterestResponse(BaseModel):
    """API response model for user interests."""

    user_id: str
    interests: List[TagSuggestionResponse]
    total_subscriptions: int
    diversity_score: int


class RecommendationResponse(BaseModel):
    """API response model for channel recommendations."""

    user_id: str
    recommended_channels: List[str]
    reasoning: str


class TopicAnalysisResponse(BaseModel):
    """API response model for topic analysis."""

    total_topics: int
    top_topics: List[dict]
    total_channels_analyzed: int


class AnalysisStatusResponse(BaseModel):
    """API response model for analysis status."""

    status: str
    message: str
    channels_processed: int
    topics_discovered: int


# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Initialize ML services (singleton pattern)
_tag_generator = None
_topic_modeler = None
_feature_extractor = None
_recommendation_engine = None


def get_tag_generator() -> TagGenerator:
    """Get or create tag generator instance."""
    global _tag_generator
    if _tag_generator is None:
        _tag_generator = TagGenerator()
    return _tag_generator


def get_topic_modeler() -> TopicModeler:
    """Get or create topic modeler instance."""
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = TopicModeler()
    return _topic_modeler


def get_feature_extractor() -> FeatureExtractor:
    """Get or create feature extractor instance."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor


def get_recommendation_engine() -> RecommendationEngine:
    """Get or create recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


@router.post("/analyze/channel/{channel_id}", response_model=ChannelTaggingResponse)
async def analyze_channel(
    channel_id: str,
    db: Session = Depends(get_db),
    tag_generator: TagGenerator = Depends(get_tag_generator),
):
    """
    Analyze a channel and generate intelligent tags.

    This endpoint combines content analysis, collaborative filtering,
    and topic modeling to generate comprehensive tags for a channel.
    """
    try:
        # Generate channel tags
        channel_tagging = tag_generator.generate_channel_tags(db, channel_id)

        # Save tags to database
        tag_generator.save_channel_tags(db, channel_tagging)

        # Convert to response format
        tag_responses = [
            TagSuggestionResponse(
                name=tag.name,
                category=tag.category,
                confidence=tag.confidence,
                source=tag.source,
                reasoning=tag.reasoning,
            )
            for tag in channel_tagging.suggested_tags
        ]

        return ChannelTaggingResponse(
            channel_id=channel_id,
            suggested_tags=tag_responses,
            primary_category=channel_tagging.primary_category,
            confidence_score=channel_tagging.confidence_score,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/interests/{user_id}", response_model=UserInterestResponse)
async def get_user_interests(
    user_id: str,
    db: Session = Depends(get_db),
    tag_generator: TagGenerator = Depends(get_tag_generator),
):
    """
    Analyze user interests based on subscription patterns.

    This endpoint analyzes a user's subscription history to generate
    personalized interest tags and content preferences.
    """
    try:
        # Generate user interest tags
        interest_tags = tag_generator.generate_user_interest_tags(db, user_id)

        # Save interests to database
        tag_generator.save_user_interests(db, user_id, interest_tags)

        # Convert to response format
        interest_responses = [
            TagSuggestionResponse(
                name=tag.name,
                category=tag.category,
                confidence=tag.confidence,
                source=tag.source,
                reasoning=tag.reasoning,
            )
            for tag in interest_tags
        ]

        # Get subscription count for context
        from backend.src.enoro.database.models.channel import UserSubscription

        subscription_count = (
            db.query(UserSubscription)
            .filter(UserSubscription.user_id == user_id)
            .count()
        )

        # Calculate diversity score
        categories = set(tag.category for tag in interest_tags)
        diversity_score = len(categories)

        return UserInterestResponse(
            user_id=user_id,
            interests=interest_responses,
            total_subscriptions=subscription_count,
            diversity_score=diversity_score,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Interest analysis failed: {str(e)}"
        )


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = 10,
    strategy: str = "auto",
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Generate personalized channel recommendations for a user.

    This endpoint uses the hybrid recommendation engine that combines
    collaborative filtering, content-based filtering, and popularity-based
    recommendations to suggest channels the user might be interested in.

    Strategies:
    - auto: Automatically choose the best strategy based on user data
    - hybrid: Combine all recommendation approaches
    - collaborative: Use collaborative filtering only
    - content: Use content-based filtering only
    - popularity: Use popularity-based recommendations only
    """
    try:
        # Initialize recommendation engine if not ready
        if not rec_engine.is_ready():
            if not rec_engine.initialize(db):
                return RecommendationResponse(
                    user_id=user_id,
                    recommended_channels=[],
                    reasoning="Recommendation system is not ready. Insufficient data for training.",
                )

        # Get detailed recommendations
        recommendations = rec_engine.get_recommendations(db, user_id, limit, strategy)

        if not recommendations:
            reasoning = (
                f"No recommendations available using {strategy} strategy. "
                "User may need more subscription data or system needs more training data."
            )
            recommended_channels = []
        else:
            reasoning = f"Recommendations generated using {strategy} strategy based on user preferences and behavior"
            recommended_channels = [rec["channel_id"] for rec in recommendations]

        return RecommendationResponse(
            user_id=user_id,
            recommended_channels=recommended_channels,
            reasoning=reasoning,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        )


@router.get("/recommendations/{user_id}/detailed")
async def get_detailed_recommendations(
    user_id: str,
    limit: int = 10,
    strategy: str = "auto",
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Get detailed channel recommendations with explanations and metadata.

    Returns comprehensive recommendation data including:
    - Channel information (name, subscriber count, etc.)
    - Recommendation scores and confidence
    - Explanations for why each channel was recommended
    - Source algorithm that generated the recommendation
    """
    try:
        # Initialize recommendation engine if not ready
        if not rec_engine.is_ready():
            if not rec_engine.initialize(db):
                return {
                    "user_id": user_id,
                    "recommendations": [],
                    "message": "Recommendation system is not ready. Insufficient data for training.",
                    "total_recommendations": 0,
                }

        # Get detailed recommendations
        recommendations = rec_engine.get_recommendations(db, user_id, limit, strategy)

        return {
            "user_id": user_id,
            "strategy_used": strategy,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": rec_engine.last_training_time.isoformat()
            if rec_engine.last_training_time
            else None,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detailed recommendation generation failed: {str(e)}",
        )


@router.get("/similar-channels/{channel_id}")
async def get_similar_channels(
    channel_id: str,
    limit: int = 10,
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Get channels similar to a specific channel based on content and features.

    Useful for:
    - Finding channels with similar content
    - Channel discovery based on known preferences
    - Content analysis and categorization
    """
    try:
        # Initialize recommendation engine if not ready
        if not rec_engine.is_ready():
            if not rec_engine.initialize(db):
                return {
                    "channel_id": channel_id,
                    "similar_channels": [],
                    "message": "Recommendation system is not ready. Insufficient data for training.",
                }

        # Get similar channels
        similar_channels = rec_engine.get_similar_channels(db, channel_id, limit)

        return {
            "channel_id": channel_id,
            "similar_channels": similar_channels,
            "total_similar": len(similar_channels),
            "algorithm": "content-based similarity",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Similar channel search failed: {str(e)}"
        )


@router.post("/recommendations/initialize")
async def initialize_recommendation_engine(
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Initialize/retrain the recommendation engine with current data.

    This endpoint should be called:
    - When setting up the system for the first time
    - After significant data updates (new channels, users, etc.)
    - Periodically to keep models fresh with new data
    """
    try:
        success = rec_engine.initialize(db)

        if success:
            health_status = rec_engine.get_health_status()
            return {
                "status": "success",
                "message": "Recommendation engine initialized successfully",
                "health": health_status,
                "ready_for_recommendations": rec_engine.is_ready(),
            }
        else:
            return {
                "status": "partial_success",
                "message": "Recommendation engine initialization completed with some issues",
                "health": rec_engine.get_health_status(),
                "ready_for_recommendations": rec_engine.is_ready(),
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation engine initialization failed: {str(e)}",
        )


@router.get("/recommendations/stats/{user_id}")
async def get_recommendation_stats(
    user_id: str,
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Get comprehensive statistics about recommendations for a user.

    Includes:
    - User profile information
    - Recommendation model status
    - Algorithm performance metrics
    - Data availability and quality indicators
    """
    try:
        stats = rec_engine.get_recommendation_stats(db, user_id)
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation stats retrieval failed: {str(e)}"
        )


@router.post("/train/topics", response_model=AnalysisStatusResponse)
async def train_topic_models(
    db: Session = Depends(get_db),
    topic_modeler: TopicModeler = Depends(get_topic_modeler),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor),
):
    """
    Train topic models on current channel data.

    This endpoint should be run periodically to update topic models
    with new channel data and improve recommendation quality.
    """
    try:
        # Fit feature extraction models
        from backend.src.enoro.database.models.channel import Channel

        channels = db.query(Channel).all()

        if len(channels) < 10:
            return AnalysisStatusResponse(
                status="insufficient_data",
                message="Need at least 10 channels to train models",
                channels_processed=len(channels),
                topics_discovered=0,
            )

        # Train feature extractor
        feature_extractor.fit_models(channels)

        # Train topic models
        topics = topic_modeler.fit_topic_models(db)

        # Save topics to database
        topic_modeler.save_topics_to_db(db, topics)

        return AnalysisStatusResponse(
            status="success",
            message=f"Successfully trained models on {len(channels)} channels",
            channels_processed=len(channels),
            topics_discovered=len(topics),
        )

    except ValueError as e:
        return AnalysisStatusResponse(
            status="error", message=str(e), channels_processed=0, topics_discovered=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.get("/topics/analysis", response_model=TopicAnalysisResponse)
async def get_topic_analysis(db: Session = Depends(get_db)):
    """
    Get current topic analysis results.

    This endpoint returns information about discovered topics
    and their associated channels.
    """
    try:
        from backend.src.enoro.database.models.tags import TopicCluster

        # Get all topics
        topics = db.query(TopicCluster).all()

        # Format top topics
        top_topics = []
        for topic in topics[:10]:  # Top 10 topics
            try:
                keywords = topic.keywords if isinstance(topic.keywords, list) else []
                if isinstance(topic.keywords, str):
                    import json

                    keywords = json.loads(topic.keywords)
            except Exception:
                keywords = []

            top_topics.append(
                {
                    "name": topic.name,
                    "keywords": keywords[:5],  # Top 5 keywords
                    "channel_count": topic.channel_count,
                    "coherence_score": topic.coherence_score,
                }
            )

        # Count total analyzed channels
        from backend.src.enoro.database.models.channel import Channel

        total_channels = db.query(Channel).count()

        return TopicAnalysisResponse(
            total_topics=len(topics),
            top_topics=top_topics,
            total_channels_analyzed=total_channels,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic analysis failed: {str(e)}")


@router.post("/analyze/batch")
async def analyze_all_channels(
    limit: Optional[int] = None,
    db: Session = Depends(get_db),
    tag_generator: TagGenerator = Depends(get_tag_generator),
):
    """
    Analyze all channels in the database and generate tags.

    This is a batch operation that should be run periodically
    to keep channel tags up to date.
    """
    try:
        from backend.src.enoro.database.models.channel import Channel

        # Get channels to analyze
        query = db.query(Channel)
        if limit:
            query = query.limit(limit)
        channels = query.all()

        if not channels:
            return {
                "status": "no_data",
                "message": "No channels found to analyze",
                "processed": 0,
            }

        # Analyze each channel
        processed = 0
        errors = 0

        for channel in channels:
            try:
                # Generate tags for channel
                channel_tagging = tag_generator.generate_channel_tags(
                    db, str(channel.id)
                )

                # Save tags
                tag_generator.save_channel_tags(db, channel_tagging)

                processed += 1

            except Exception as e:
                print(f"Error analyzing channel {channel.id}: {e}")
                errors += 1

        return {
            "status": "completed",
            "message": f"Analyzed {processed} channels successfully",
            "processed": processed,
            "errors": errors,
            "total_channels": len(channels),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/status")
async def get_ml_status(
    db: Session = Depends(get_db),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """
    Get overall ML system status and statistics.
    """
    try:
        from backend.src.enoro.database.models.channel import Channel
        from backend.src.enoro.database.models.tags import (
            ContentTag,
            ChannelTag,
            TopicCluster,
        )

        # Count various entities
        total_channels = db.query(Channel).count()
        total_tags = db.query(ContentTag).count()
        tagged_channels = db.query(ChannelTag.channel_id).distinct().count()
        total_topics = db.query(TopicCluster).count()

        # Calculate coverage
        tag_coverage = (
            (tagged_channels / total_channels * 100) if total_channels > 0 else 0
        )

        # Get recommendation system health
        rec_health = rec_engine.get_health_status()

        return {
            "system_status": "operational",
            "statistics": {
                "total_channels": total_channels,
                "tagged_channels": tagged_channels,
                "tag_coverage_percent": round(tag_coverage, 1),
                "total_tags": total_tags,
                "discovered_topics": total_topics,
            },
            "recommendations": {
                "ready": rec_engine.is_ready(),
                "engine_status": rec_health["status"],
                "models_trained": rec_health["models_trained"],
                "last_training": rec_health["last_training"],
                "cache_size": rec_health["cache_size"],
                "min_data_threshold": "10 tagged channels required",
            },
            "content_analysis": {
                "ready": total_topics > 0 and tagged_channels > 10,
                "tag_coverage": round(tag_coverage, 1),
                "topics_discovered": total_topics,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

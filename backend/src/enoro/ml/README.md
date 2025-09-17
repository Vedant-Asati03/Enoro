# ML Package Structure

This directory contains all machine learning components for Enoro, organized for scalability and maintainability.

## Directory Structure

```
ml/
├── content_analysis/          # Text processing and content understanding
│   ├── __init__.py
│   ├── feature_extraction.py  # Text feature extraction and NLP
│   ├── topic_modeling.py      # LDA/NMF topic discovery
│   └── tag_generation.py      # Intelligent tag generation engine
│
├── recommendations/           # Recommendation systems (future)
│   ├── __init__.py
│   ├── collaborative_filtering.py    # User-based & item-based CF
│   ├── content_based.py              # Content-based recommendations
│   ├── matrix_factorization.py       # SVD, NMF for recommendations
│   ├── deep_learning.py              # Neural collaborative filtering
│   └── hybrid.py                     # Hybrid recommendation systems
│
├── shared/                    # Common utilities and tools
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── model_utils.py         # Model saving/loading utilities
│   └── evaluation.py          # Metrics and evaluation tools
│
├── models/                    # Trained model storage
│   ├── content_analysis/      # Saved content analysis models
│   ├── recommendations/       # Saved recommendation models
│   └── cache/                 # Temporary model cache
│
└── __init__.py               # Main package interface
```

## Components Overview

### Content Analysis (`content_analysis/`)

**Purpose**: Understand and categorize YouTube content through text analysis.

- **FeatureExtractor**: Extracts keywords, entities, sentiment from channel descriptions
- **TopicModeler**: Discovers content themes using LDA/NMF algorithms  
- **TagGenerator**: Combines multiple signals to generate intelligent content tags

**Key Features**:

- Text preprocessing and cleaning
- Keyword and entity extraction
- Topic modeling and clustering
- Multi-source tag generation (content + collaborative + topics)
- Confidence scoring and ranking

### Recommendations (`recommendations/`)

**Purpose**: Generate personalized channel recommendations for users.

**Planned Components**:

- **Collaborative Filtering**: User-based and item-based recommendations
- **Content-Based**: Recommendations based on content similarity
- **Matrix Factorization**: SVD, NMF for scalable recommendations
- **Deep Learning**: Neural collaborative filtering
- **Hybrid Systems**: Combining multiple recommendation approaches

### Shared Utilities (`shared/`)

**Purpose**: Common functionality used across ML components.

- **DataPreprocessor**: Data cleaning, normalization, feature encoding
- **ModelManager**: Model saving, loading, versioning, and cleanup
- **MetricsCalculator**: Evaluation metrics for recommendations and classification

**Key Features**:

- User-item matrix creation
- Train/test splitting
- Model persistence with metadata
- Performance tracking
- Recommendation metrics (Precision@K, Recall@K, NDCG)

### Models Directory (`models/`)

**Purpose**: Storage for trained ML models and cached data.

- Organized by component type
- Automatic versioning and metadata tracking
- Performance history tracking
- Cleanup utilities for old versions

## Usage Examples

### Content Analysis

```python
from backend.src.enoro.ml import TagGenerator, TopicModeler

# Generate tags for a channel
tag_generator = TagGenerator()
tagging = tag_generator.generate_channel_tags(db, channel_id)

# Train topic models
topic_modeler = TopicModeler()
topics = topic_modeler.fit_topic_models(db)
```

### Data Processing

```python
from backend.src.enoro.ml import DataPreprocessor

# Create user-item matrix
preprocessor = DataPreprocessor()
matrix, users, channels = preprocessor.get_user_subscription_matrix(db)
```

### Model Management

```python
from backend.src.enoro.ml import ModelManager

# Save a trained model
manager = ModelManager()
manager.save_model(model, "tag_classifier", version="1.0")

# Load a model
model = manager.load_model("tag_classifier")
```

## API Endpoints

The ML functionality is exposed through REST API endpoints in `/ml/`:

- `POST /ml/analyze/channel/{id}` - Generate tags for a channel
- `GET /ml/interests/{user_id}` - Analyze user interests
- `GET /ml/recommendations/{user_id}` - Get personalized recommendations
- `POST /ml/train/topics` - Train topic models
- `GET /ml/status` - System status and statistics

## Future Roadmap

### Phase 1: Content Analysis ✅

- [x] Feature extraction from text
- [x] Topic modeling and clustering
- [x] Intelligent tag generation
- [x] API endpoints

### Phase 2: Basic Recommendations (Next)

- [ ] Collaborative filtering implementation
- [ ] Content-based recommendations
- [ ] User similarity calculation
- [ ] Recommendation API endpoints

### Phase 3: Advanced Recommendations

- [ ] Matrix factorization techniques
- [ ] Deep learning models
- [ ] Hybrid recommendation systems
- [ ] Real-time recommendation updates

### Phase 4: Production Optimization

- [ ] Model serving optimization
- [ ] Caching and performance tuning
- [ ] A/B testing framework
- [ ] Online learning capabilities

## Development Guidelines

1. **Modular Design**: Each component should be independently testable
2. **Clear Interfaces**: Well-defined APIs between components
3. **Performance**: Optimize for speed and memory usage
4. **Scalability**: Design for growing data and user base
5. **Monitoring**: Track model performance and data drift
6. **Documentation**: Keep this README and code comments updated

## Dependencies

Core ML libraries:

- `scikit-learn` - Traditional ML algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `nltk` - Natural language processing

Future additions:

- `tensorflow` or `pytorch` - Deep learning
- `implicit` - Fast collaborative filtering
- `surprise` - Recommendation system library

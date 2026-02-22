# ML Classification System - Setup & Development Guide

## Quick Setup (5 minutes)

### Prerequisites
- Python 3.11+
- pip
- Git
- PostgreSQL (or use Neon serverless)

### 1. Clone and Setup

```bash
# Clone repository
git clone <repo-url>
cd ml-classification-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
# DATABASE_URL (PostgreSQL connection string)
# WANDB_API_KEY (optional)
# MODEL_PATH (usually model_artifacts/model.joblib)
```

### 3. Run Locally

**Option A: Without Docker (3 terminals)**

```bash
# Terminal 1: Start FastAPI backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/app.py

# Terminal 3 (optional): Train new model
python scripts/train.py
```

Access:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Option B: With Docker Compose (1 command)**

```bash
# Start entire stack (API, Frontend, Database, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d

# Stop
docker-compose -f docker/docker-compose.yml down
```

Access:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Development Workflow

### Making Changes

1. Create a branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make changes to code

3. Format and lint:
   ```bash
   # Format code
   black backend frontend tests
   
   # Check formatting
   flake8 backend frontend tests
   
   # Lint
   pylint backend/**/*.py
   ```

4. Run tests:
   ```bash
   pytest tests/ -v --cov=backend
   ```

5. Commit and push:
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

6. Create Pull Request on GitHub

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=html

# Run specific test
pytest tests/test_all.py::TestMLPipeline::test_prediction -v

# Run tests matching pattern
pytest tests/ -k "prediction" -v

# Run with markers
pytest tests/ -m unit -v
```

### Code Quality

```bash
# Format all Python files
black backend frontend tests

# Check with Flake8
flake8 backend frontend tests

# Full Pylint report
pylint backend/**/*.py --rcfile=pylint.rc

# Type checking
mypy backend/**/*.py
```

### Debugging

```bash
# Run backend with auto-reload and debug logging
LOG_LEVEL=debug uvicorn backend.main:app --reload

# Run tests with verbose output and breakpoints
pytest tests/ -v -s  # -s shows print statements

# Use pdb for debugging
import pdb; pdb.set_trace()
```

## Project Structure Guide

```
backend/
├── main.py          # FastAPI application
├── ml_pipeline.py   # ML training and inference
└── data_manager.py  # Database operations

frontend/
└── app.py          # Streamlit UI

tests/
└── test_all.py     # Comprehensive test suite

scripts/
├── train.py        # Model training
└── generate_report.py  # Documentation

docker/
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml

monitoring/
└── prometheus.yml

.github/workflows/
├── backend.yml
└── frontend.yml

docs/
├── DEPLOYMENT.md
└── reports/MLOps_Comprehensive_Report.md
```

## API Development

### Adding New Endpoints

1. In `backend/main.py`:

```python
@app.post("/new-endpoint", tags=["New"])
def new_endpoint(input_data: SomeModel):
    """Description of endpoint."""
    # Your logic here
    return response
```

2. Define Pydantic models in same file

3. Add tests in `tests/test_all.py`

4. Run tests: `pytest tests/test_all.py -v`

### Using Database

```python
from backend.data_manager import DataManager

# Initialize
dm = DataManager(DATABASE_URL)

# Get data
df = dm.get_training_data()

# Save prediction
dm.save_prediction([features], prediction, confidence)
```

## ML Pipeline Development

### Training a New Model

```python
from backend.ml_pipeline import MLPipeline, load_and_prepare_data

# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data("data/dataset.csv")

# Create pipeline
pipeline = MLPipeline()
pipeline.create_pipeline()

# Tune hyperparameters
pipeline.tune_hyperparameters(X_train, y_train, search_type="random")

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)

# Save
pipeline.save_model("model_artifacts/model.joblib")
```

### Experiments with W&B

```bash
# Login
wandb login

# Set API key
export WANDB_API_KEY=your_key

# Train (automatically logs to W&B)
python scripts/train.py

# View experiments
https://wandb.ai/your-username/ml-classification
```

## Deployment

### Local Docker

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Run
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs -f backend

# Stop
docker-compose -f docker/docker-compose.yml down
```

### Render Cloud

1. Push to GitHub main branch
2. GitHub Actions automatically:
   - Runs tests
   - Builds Docker image
   - Deploys to Render
3. Monitor at Render dashboard

## Monitoring

### Prometheus

```bash
# Access at http://localhost:9090
# Query metrics: predictions_total, prediction_latency_seconds
```

### Grafana

```bash
# Access at http://localhost:3000
# Username: admin
# Password: admin
# Add Prometheus data source: http://prometheus:9090
```

## Database

### Using Neon (Serverless PostgreSQL)

1. Go to https://console.neon.tech
2. Create project
3. Get connection string
4. Set DATABASE_URL=connection_string in .env

### Local PostgreSQL

```bash
# Using Docker
docker run -d \
  -e POSTGRES_USER=mlops_user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mlops_db \
  -p 5432:5432 \
  postgres:15-alpine
```

## Troubleshooting

### API won't start
```bash
# Check model file exists
ls -la model_artifacts/model.joblib

# Check environment variables
echo $DATABASE_URL
echo $MODEL_PATH

# Check logs
# In Docker: docker logs mlops-backend
# In terminal: Look at startup errors
```

### Frontend can't connect to API
```bash
# Check API is running
curl http://localhost:8000/health

# Check API_URL in frontend
# Run with debug logging:
streamlit run frontend/app.py --logger.level=debug
```

### Tests failing
```bash
# Check dependencies installed
pip install -r requirements.txt

# Run single test with verbose output
pytest tests/test_all.py::TestMLPipeline::test_prediction -v -s

# Check Python version
python --version  # Should be 3.11+
```

### Database connection error
```bash
# Verify DATABASE_URL format
# postgresql://user:password@host:port/database

# Test connection
psql $DATABASE_URL

# For Neon, use:
# postgresql://user:password@ep-xxx.neon.tech/database
```

## Performance Optimization

### API Optimization
```python
# Add caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model():
    return joblib.load(MODEL_PATH)

# Connection pooling
pool = create_engine(db_url, pool_size=20, max_overflow=10)
```

### Frontend Optimization
- Use Streamlit caching: @st.cache_resource
- Minimize API calls
- Paginate large results

### Model Optimization
- Convert to ONNX format
- Use model quantization
- Implement batch processing

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Format and lint code
6. Create Pull Request

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Docker Guide](https://docs.docker.com/)
- [Weights & Biases](https://docs.wandb.ai/)
- [PostgreSQL](https://www.postgresql.org/docs/)

## Getting Help

- Check [README.md](README.md)
- See [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Read [MLOps Report](docs/reports/MLOps_Comprehensive_Report.md)
- GitHub Issues: [repo issues]
- Email: support@example.com

---

**Happy developing!** 🚀

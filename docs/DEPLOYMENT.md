# Deployment Guide

This guide covers deploying the ML Classification system to production on Render.

## Table of Contents

1. [Pre-deployment Checklist](#pre-deployment-checklist)
2. [Render Backend Deployment](#render-backend-deployment)
3. [Render Frontend Deployment](#render-frontend-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Monitoring in Production](#monitoring-in-production)
7. [Scaling Considerations](#scaling-considerations)

## Pre-deployment Checklist

- [ ] All tests passing locally (`pytest tests/ -v`)
- [ ] Code formatted (`black .`)
- [ ] Linting passed (`flake8 . && pylint backend/**/*.py`)
- [ ] Model artifacts present in `/model_artifacts/`
- [ ] GitHub repository created and pushed
- [ ] Render account created
- [ ] PostgreSQL database provisioned (Neon or Render)
- [ ] W&B account setup (optional)
- [ ] Secrets configured

## Render Backend Deployment

### 1. Create Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ml-classification-backend`
   - **Environment**: `Python 3.11`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
   - **Plan**: Starter ($7/month) or higher

### 2. Set Environment Variables

In Render dashboard, set:
```
DATABASE_URL=postgresql://user:password@host/database
MODEL_PATH=model_artifacts/model.joblib
WANDB_API_KEY=your_wandb_key
LOG_LEVEL=info
```

### 3. Configure Health Check

- **HTTP Path**: `/health`
- **Check Interval**: 30 seconds
- **Timeout**: 10 seconds

### 4. Deploy

Render auto-deploys when you push to main branch. View logs in Render dashboard.

## Render Frontend Deployment

### 1. Create Web Service

1. Click "New +" → "Web Service"
2. Configure:
   - **Name**: `ml-classification-frontend`
   - **Environment**: `Python 3.11`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run frontend/app.py --server.port=10000 --server.address=0.0.0.0`

### 2. Set Environment Variables

```
API_URL=https://ml-classification-backend.onrender.com
STREAMLIT_SERVER_PORT=10000
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Configure

- Allow up to 120s for startup
- Use largest instance size for better performance
- Enable Python version 3.11

## Environment Configuration

### Production `.env`

```bash
# Database
DATABASE_URL=postgresql://neon_user:password@neon_host/db_name

# Model
MODEL_PATH=model_artifacts/model.joblib

# API
API_PORT=10000
LOG_LEVEL=info

# Monitoring
WANDB_API_KEY=your_api_key
PROMETHEUS_PORT=9090

# Frontend
FRONTEND_URL=https://ml-classification-frontend.onrender.com
API_URL=https://ml-classification-backend.onrender.com

# Security
ALLOWED_ORIGINS=https://ml-classification-frontend.onrender.com
```

## Database Setup

### Option 1: Neon Postgres (Recommended)

1. Go to [Neon Console](https://console.neon.tech)
2. Create project
3. Get connection string
4. Set `DATABASE_URL` in Render

### Option 2: Render Postgres

1. In Render, create PostgreSQL database
2. Get connection info
3. Use as `DATABASE_URL`

### Initialize Database

After deploying backend, tables auto-create on first connection.

## Monitoring in Production

### Prometheus Metrics

Access metrics at: `https://your-backend.onrender.com/metrics`

Monitor:
- `predictions_total`: Prediction count by class
- `prediction_latency_seconds`: Average prediction time
- `requests_total`: API request count

### Logging

- Backend logs available in Render dashboard
- Set `LOG_LEVEL=info` for production

### Health Checks

- Endpoint: `/health`
- Check interval: 30 seconds
- Auto-restart on failure

## Scaling Considerations

### Horizontal Scaling
- Render Pro: Auto-scaling available
- Load balancing: Handled by Render
- Session management: API is stateless

### Performance Optimization
1. **Caching**:
   ```python
   from functools import lru_cache
   @lru_cache(maxsize=128)
   def get_model():
       return joblib.load(MODEL_PATH)
   ```

2. **Connection Pooling**:
   ```python
   pool = create_engine(db_url, pool_size=20)
   ```

3. **Model Quantization**: Convert model to ONNX for faster inference

### Cost Optimization
- Use shared CPU instances for development
- Reserve instances for production
- Use CDN for static assets
- Archive old predictions periodically

## CI/CD Integration

GitHub Actions automatically:
1. Runs tests on push
2. Builds Docker images
3. Deploys to Render on main branch

### Deployment Workflow

```yaml
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy Backend
        run: |
          curl -d {} https://api.render.com/deploy/...
```

## Troubleshooting

### Backend Won't Start
- Check logs: `Render Dashboard → Logs`
- Verify `MODEL_PATH` exists
- Ensure `DATABASE_URL` is valid

### Frontend Can't Connect
- Verify `API_URL` in environment
- Check CORS settings in backend
- Ensure backend is running

### Slow Predictions
- Check database connection
- Profile model inference time
- Consider model optimization

### High Memory Usage
- Reduce batch size
- Limit model caching
- Monitor with Prometheus

## Rollback Procedure

1. Go to Render dashboard
2. Find service
3. Click "Deployments" tab
4. Select previous version
5. Click "Deploy"

## Security Checklist

- [ ] Use HTTPS (default on Render)
- [ ] Rotate API keys regularly
- [ ] Use environment variables for secrets
- [ ] Enable firewall rules
- [ ] Monitor access logs
- [ ] Implement rate limiting
- [ ] Use VPC (Pro plans)
- [ ] Regular security audits

## Contact & Support

- Render Support: support@render.com
- GitHub Issues: [Repository Issues]
- Discord: [Community]

---

**Last Updated**: February 2024

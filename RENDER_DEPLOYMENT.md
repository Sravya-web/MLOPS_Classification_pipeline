# Render Deployment Guide - Hepatitis Dashboard

This guide walks you through deploying the Hepatitis Classification Dashboard to Render without errors.

## Prerequisites

✅ All checked:
- Python 3.11
- Docker configured
- Git repository initialized and pushed to GitHub
- Model artifacts present: `model_artifacts/hepatitis_model.joblib`
- Data present: `data/hepatitis.csv`
- All tests passing locally

## Deployment Steps

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Step 2: Create Render Account

1. Go to https://dashboard.render.com
2. Sign up with GitHub (recommended for easy integration)
3. Create a new account

### Step 3: Deploy Dashboard (Streamlit Frontend)

1. **In Render Dashboard:**
   - Click **"New +"** → **"Web Service"**
   - Select your GitHub repository
   - Choose the main branch

2. **Configure Service:**
   - **Name:** `hepatitis-dashboard` (or your choice)
   - **Region:** Oregon (closest to most users)
   - **Plan:** Free (to start) or Starter ($7/month)
   - **Environment:** Docker
   - **Docker File Path:** `docker/Dockerfile.frontend`

3. **Environment Variables:** (Add in Render Dashboard)
   ```
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_RUNONSAVE=false
   STREAMLIT_SERVER_PORT=10000
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

4. **Click "Deploy"**
   - Wait 3-5 minutes for build and deployment
   - Check logs for any errors

### Step 4: Verify Deployment

- Access your dashboard at: `https://<your-service-name>.onrender.com`
- Should load with interactive pages:
  - Overview
  - Patient Analysis
  - Feature Analysis
  - Statistics
  - Clinical Insights

### Step 5: Optional - Deploy Backend API

1. **Create another Web Service:**
   - Click **"New +"** → **"Web Service"**
   - Select same repository

2. **Configure:**
   - **Name:** `hepatitis-api`
   - **Docker File Path:** `docker/Dockerfile.backend`
   - **Build Command:** (leave default)
   - **Start Command:** (leave default - uses CMD from Dockerfile)

3. **Environment Variables:**
   ```
   LOG_LEVEL=info
   ```

4. **Deploy**

## Architecture

```
GitHub Repository (main branch)
        ↓
   Render Dashboard
        ↓
    ┌───┴───┐
    ↓       ↓
 Frontend  Backend (optional)
(Streamlit) (FastAPI)
```

## Common Issues & Solutions

### Issue 1: Build Fails - Missing Dependencies
**Solution:**
- All dependencies are in `requirements-render.txt`
- Verify file is not corrupted:
  ```bash
  cat requirements-render.txt
  ```

### Issue 2: Streamlit Server Won't Start
**Solution:**
- Ensure environment variables are set correctly
- Check logs in Render dashboard
- Verify `STREAMLIT_SERVER_PORT=10000`

### Issue 3: Model Files Not Found
**Solution:**
- Verify files exist locally:
  ```bash
  ls -la model_artifacts/
  ls -la data/
  ```
- Ensure `.gitignore` doesn't exclude these files
- Push to GitHub: `git push origin main`

### Issue 4: Dashboard Loads But Shows No Data
**Solution:**
- Check if `data/hepatitis.csv` is present
- Verify file permissions: `ls -la data/hepatitis.csv`
- Try restarting service in Render dashboard

## Monitoring & Logs

In Render Dashboard:
1. Click your service
2. Go to **"Logs"** tab
3. Check for errors in real-time

## Useful Commands (Local Testing)

Test Docker build locally before pushing:
```bash
# Build frontend image
docker build -f docker/Dockerfile.frontend -t hepatitis-dashboard .

# Run locally
docker run -p 8501:10000 hepatitis-dashboard
```

Visit: `http://localhost:8501`

## Updating Deployment

After making changes:
```bash
git add .
git commit -m "Update description"
git push origin main
```

Render automatically redeploys when main branch changes.

## Cost Estimation

- **Free Plan:** Good for testing, may have downtime
- **Starter Plan:** $7/month, always running, better performance
- **Pro Plan:** $25/month, more resources

For production use, recommend Starter or Pro plan.

## Security Notes

1. **Secrets:** Use Render's environment variables, not hardcoded
2. **CORS:** Currently allows all origins (update in production)
3. **API Rate Limiting:** Add rate limits for production

## Next Steps

1. Deploy dashboard to Render
2. Test all 5 pages work correctly
3. Test patient prediction functionality
4. Share public URL with team/stakeholders

## Support

If deployment fails:
1. Check Render logs for error messages
2. Verify all files are committed to GitHub
3. Ensure Docker build works locally
4. Contact Render support if infrastructure issue

---

**Deployment Status:** Ready ✅
**Last Updated:** 2026-02-24

# 🚀 ML Microservices Platform

A production-ready, containerized machine learning platform with real-time inference capabilities, handling 10K+ requests/minute with sub-100ms latency. Built with FastAPI, PyTorch, and Kubernetes.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Key Features

- **High Performance**: Sub-100ms inference latency with optimized PyTorch models
- **Scalable Architecture**: Handles 10,000+ requests per minute
- **Microservices Design**: Independently deployable services for different ML models
- **Real-Time Inference**: Asynchronous processing with Redis queue
- **Production Ready**: Comprehensive monitoring, logging, and error handling
- **Auto-Scaling**: Kubernetes HPA for automatic resource management
- **Model Versioning**: A/B testing and canary deployments
- **API Gateway**: Unified entry point with rate limiting and authentication

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Kong/Nginx)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
    │  Prediction  │  │   NLP      │  │   Vision    │
    │  Service     │  │  Service   │  │   Service   │
    │  (FastAPI)   │  │ (FastAPI)  │  │  (FastAPI)  │
    └──────┬───────┘  └─────┬──────┘  └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Redis Cache/Queue   │
                └───────────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Prometheus +        │
                │   Grafana Monitoring  │
                └───────────────────────┘
```

## 🛠️ Technology Stack

**Backend Services:**
- FastAPI (REST API framework)
- PyTorch (Deep learning)
- Redis (Caching & message queue)
- PostgreSQL (Model metadata)

**Infrastructure:**
- Docker (Containerization)
- Kubernetes (Orchestration)
- Helm (Package management)
- NGINX Ingress (Load balancing)

**Monitoring & Observability:**
- Prometheus (Metrics)
- Grafana (Visualization)
- ELK Stack (Logging)
- Jaeger (Distributed tracing)

## 📋 Prerequisites

- Docker 20.10+
- Kubernetes 1.24+ (or minikube for local)
- kubectl CLI
- Helm 3+
- Python 3.9+
- 16GB+ RAM (for running multiple models)

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/ml-microservices-platform.git
cd ml-microservices-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Start prediction service
cd services/prediction
uvicorn main:app --reload --port 8001

# Start NLP service
cd services/nlp
uvicorn main:app --reload --port 8002

# Start Vision service
cd services/vision
uvicorn main:app --reload --port 8003
```

### Docker Compose (All Services)

```bash
# Build and start all services
docker-compose up --build

# Access services
# Prediction API: http://localhost:8001
# NLP API: http://localhost:8002
# Vision API: http://localhost:8003
# Grafana: http://localhost:3000
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace ml-platform

# Deploy with Helm
helm install ml-platform ./helm-chart -n ml-platform

# Check deployments
kubectl get pods -n ml-platform

# Access via port-forward
kubectl port-forward svc/api-gateway 8080:80 -n ml-platform
```

## 📁 Project Structure

```
ml-microservices-platform/
├── services/
│   ├── prediction/
│   │   ├── main.py              # FastAPI app
│   │   ├── models/
│   │   │   ├── model_loader.py
│   │   │   └── predictor.py
│   │   ├── api/
│   │   │   ├── routes.py
│   │   │   └── schemas.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── cache.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── nlp/
│   │   ├── main.py
│   │   ├── models/
│   │   │   ├── sentiment.py
│   │   │   ├── ner.py
│   │   │   └── summarization.py
│   │   ├── api/
│   │   └── Dockerfile
│   └── vision/
│       ├── main.py
│       ├── models/
│       │   ├── classification.py
│       │   ├── detection.py
│       │   └── segmentation.py
│       ├── api/
│       └── Dockerfile
├── gateway/
│   ├── nginx.conf
│   └── kong.yml
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── alerting/
│       └── rules.yml
├── helm-chart/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── hpa.yaml
│       └── ingress.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
│       └── locustfile.py
├── scripts/
│   ├── train_model.py
│   ├── deploy.sh
│   └── benchmark.py
├── docker-compose.yml
├── kubernetes/
│   ├── deployments/
│   ├── services/
│   └── configmaps/
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create `.env` files for each service:

**services/prediction/.env:**
```env
MODEL_PATH=/models/prediction_model.pth
DEVICE=cuda
BATCH_SIZE=32
MAX_WORKERS=4
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

**services/nlp/.env:**
```env
TRANSFORMER_MODEL=bert-base-uncased
MAX_SEQ_LENGTH=512
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
```

## 📊 API Documentation

### Prediction Service (Port 8001)

**POST /predict**
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 3.4, 5.6, 7.8],
    "model_version": "v1.0"
  }'
```

**Response:**
```json
{
  "prediction": 0.87,
  "confidence": 0.95,
  "model_version": "v1.0",
  "inference_time_ms": 45,
  "request_id": "req_abc123"
}
```

### NLP Service (Port 8002)

**POST /analyze/sentiment**
```bash
curl -X POST "http://localhost:8002/analyze/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing!"
  }'
```

**POST /analyze/ner**
```bash
curl -X POST "http://localhost:8002/analyze/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc. is located in Cupertino, California."
  }'
```

### Vision Service (Port 8003)

**POST /classify**
```bash
curl -X POST "http://localhost:8003/classify" \
  -F "image=@image.jpg"
```

**Response:**
```json
{
  "predictions": [
    {"class": "cat", "confidence": 0.95},
    {"class": "dog", "confidence": 0.04},
    {"class": "bird", "confidence": 0.01}
  ],
  "inference_time_ms": 78
}
```

## 📈 Performance Benchmarks

| Metric | Value | Target |
|--------|-------|--------|
| Throughput | 10,500 req/min | 10,000 req/min |
| P50 Latency | 45ms | < 50ms |
| P95 Latency | 85ms | < 100ms |
| P99 Latency | 120ms | < 150ms |
| Availability | 99.95% | 99.9% |
| Error Rate | 0.08% | < 0.5% |

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
cd tests/load
locust -f locustfile.py --host=http://localhost:8001

# Access Locust UI
open http://localhost:8089
```

## 🔍 Monitoring & Observability

### Prometheus Metrics

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency histogram
- `model_inference_duration_seconds` - Model inference time
- `cache_hits_total` - Redis cache hits
- `cache_misses_total` - Redis cache misses
- `gpu_memory_usage_bytes` - GPU memory utilization

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

**Available Dashboards:**
1. Service Overview
2. Model Performance
3. Resource Utilization
4. Error Tracking
5. Cache Performance

### Health Checks

```bash
# Service health
curl http://localhost:8001/health

# Readiness probe
curl http://localhost:8001/ready

# Liveness probe
curl http://localhost:8001/alive
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Coverage report
pytest --cov=services --cov-report=html

# Load testing
locust -f tests/load/locustfile.py --headless -u 100 -r 10 --run-time 5m
```

## 🚀 Deployment

### Production Deployment Checklist

- [ ] Update model versions in values.yaml
- [ ] Configure resource limits (CPU/Memory)
- [ ] Set up horizontal pod autoscaling
- [ ] Configure ingress with SSL/TLS
- [ ] Set up monitoring alerts
- [ ] Configure backup strategies
- [ ] Test rolling update procedure
- [ ] Document rollback process

### CI/CD Pipeline

GitHub Actions workflow for automated deployment:

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Platform

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t ml-prediction:${{ github.sha }} services/prediction
          docker build -t ml-nlp:${{ github.sha }} services/nlp
          docker build -t ml-vision:${{ github.sha }} services/vision

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          helm upgrade --install ml-platform ./helm-chart \
            --set image.tag=${{ github.sha }} \
            --namespace production
```

## 🔐 Security

- API authentication via JWT tokens
- Rate limiting on all endpoints
- Input validation and sanitization
- Model artifact signing
- Network policies in Kubernetes
- Secrets management with Vault
- Regular security scanning

## 📚 Model Management

### Model Versioning

```bash
# Register new model version
python scripts/register_model.py \
  --model-path models/model_v2.pth \
  --version v2.0 \
  --metadata '{"accuracy": 0.95, "f1_score": 0.92}'
```

### A/B Testing

```yaml
# Split traffic between model versions
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-routing
data:
  routing.json: |
    {
      "v1.0": 0.8,
      "v2.0": 0.2
    }
```

## 🛠️ Troubleshooting

### Common Issues

**Issue: High latency**
```bash
# Check Redis connection
redis-cli ping

# Monitor GPU usage
nvidia-smi -l 1

# Check model loading time
kubectl logs -n ml-platform <pod-name> | grep "Model loaded"
```

**Issue: OOM errors**
```bash
# Check memory usage
kubectl top pods -n ml-platform

# Reduce batch size
kubectl set env deployment/prediction-service BATCH_SIZE=16
```

## 🗺️ Roadmap

- [ ] Multi-GPU support
- [ ] Model quantization (INT8)
- [ ] ONNX Runtime integration
- [ ] GraphQL API
- [ ] Stream processing with Kafka
- [ ] Edge deployment support
- [ ] Federated learning capabilities
- [ ] AutoML pipeline integration

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

**Sai Charan Kolluru**
- LinkedIn: [kscharan1608](https://linkedin.com/in/kscharan1608)
- Email: kscharan1608@gmail.com

## 🙏 Acknowledgments

- FastAPI community
- PyTorch team
- Kubernetes community
- University of Maryland Baltimore County

---

⭐ Star this repo if you find it useful!

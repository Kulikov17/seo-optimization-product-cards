from fastapi import FastAPI
from src.celery.utils import create_celery
from src.routes import model, users
from prometheus_fastapi_instrumentator import Instrumentator


def create_app() -> FastAPI:
    current_app = FastAPI(
        title='Products',
        version='1.0.0',
        description='Seo optimization product cards'
    )

    # Registering the routers and endpoints
    current_app.include_router(model.router, tags=['Model'])
    current_app.include_router(users.router, tags=["Users"])

    current_app.celery_app = create_celery()
    Instrumentator().instrument(current_app).expose(current_app)

    return current_app


app = create_app()
celery = app.celery_app

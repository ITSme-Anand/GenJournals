import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #use SECRET_KEY if present in environment variables, else use a default for development
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key') 
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')
    DEVICE = os.getenv('DEVICE', 'cpu') 
    
class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'
    
class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'

class TestingConfig(Config):
    TESTING = True
    CHROMA_DIR = './test_chroma_db'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
# file backend/server/server/wsgi.py
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.cxrnet.model import CXRClassifier

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    rf = CXRClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="CXRNet",
                            algorithm_object=rf,
                            algorithm_name="CXRNet",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="DongDong",
                            algorithm_description="CoatNet for CXR Image Prediction",
                            algorithm_code=inspect.getsource(CXRClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
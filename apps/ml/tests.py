from django.test import TestCase

from apps.ml.cxrnet.model import CXRClassifier
import inspect
from apps.ml.registry import MLRegistry

import matplotlib.pyplot as plt

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = plt.imread("assets/test.jpg")
        my_alg = CXRClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "CXRNet"
        algorithm_object = CXRClassifier()
        algorithm_name = "CXRNet"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "CoatNet for CXR Image Prediction"
        algorithm_code = inspect.getsource(CXRClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)

    
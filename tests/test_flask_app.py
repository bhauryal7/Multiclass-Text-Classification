import unittest
import json
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        with open('artifacts/label_classes.json') as f:
            cls.expected_classes = json.load(f)


    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Text Classification</title>', response.data)

    def test_predict_page(self):
        sample_text = "Alpro Joghurt Strawberry"
        response = self.client.post('/predict', data=dict(text=sample_text))
        
        # print("Response data:", response.data)  # Debugging step

        self.assertEqual(response.status_code, 200)
        
        self.assertTrue(
            any(label.encode() in response.data for label in self.expected_classes),
            "Response should contain one of the expected classes." 
        )

if __name__ == '__main__':
    unittest.main()
    
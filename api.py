#!/usr/bin/python
from flask import Flask, render_template, request
from flask_restx import Api, Resource, fields
import joblib
import pickle
from model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Car Price Prediction API',
    description='Car Price Prediction API')

ns = api.namespace('Car Price Prediction', 
     description='Car Price Prediction')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year model', 
    location='args')
    
parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Car Mileage', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='State', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Make', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model', 
    location='args')   
    

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceForecast (Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['Year'],args['Mileage'],args['State'],args['Make'],args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)

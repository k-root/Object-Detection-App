from flask import Flask
from flask import current_app, Flask, redirect, url_for, Blueprint, send_from_directory, render_template, request, Response
from flask_cors import CORS
import requests
import re
import json
import train
import evaluate
import predict
import predict_individual
from .flask_app import flask_app

import os
def create_app(debug=True, testing=True, config_overrides=None):
    app = Flask(__name__, static_folder='../angular/dist/')
    app.debug = debug
    app.testing = testing
    angular = Blueprint('angular', __name__,
                            template_folder='../angular/dist/')
    app.register_blueprint(angular)
    app.register_blueprint(flask_app, url_prefix='/api')
    CORS(app)
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.errorhandler(500)
    def server_error(e):
        return """
        An internal error occurred: <pre>{}</pre>
        See logs for full stacktrace.
        """.format(e), 500

    return app

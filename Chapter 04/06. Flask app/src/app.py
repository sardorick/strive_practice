from flask import Flask
from flask_restful import Resource, Api
from routes.home.route import HomeRoute, HomeRouteWithId
from utils.db import db


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
    db.init_app(app)
    db.create_all(app=app)
    api = Api(app)
    api.add_resource(HomeRoute, '/')
    api.add_resource(HomeRouteWithId, '/<string:id>')
    return app

    


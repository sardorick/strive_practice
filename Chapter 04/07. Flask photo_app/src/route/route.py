from flask_restful import Resource, Api
from flask import request
import uuid
from utils.models.user import User
from utils.db import db

data = []
class HomeRoute(Resource):
    def get(self):
        users = db.session.query(User).all()
        users = [user.to_json() for user in users]
        return {'data': users}
    def post(self):
        id = str(uuid.uuid4())
        name = request.form["name"]
        last_name = request.form["last_name"]
        email = request.form["email"]
        user = User(first_name=name, last_name=name, email=email)
        db.session.add(user)
        db.session.commit()
        return {'data': user.to_json()}

def find_object_by_id(id):
    for data_object in data:
        if data_object["id"] == id:
            return data_object
        else:
            return None

class HomeRouteWithId(Resource):
    def get(self, id):
        data_object = find_object_by_id(id)
        if (data_object):
            return {"data": data_object}
        else:
            return {"data": "not Found"}
    def put(self, id):
        data_object = find_object_by_id(id)
        if (data_object):
            data_object["name"] = request.form["name"]
            data_object["last_name"] = request.form["last_name"]
            data_object["email"] = request.form["email"]
            return {"data": data_object} 
        else:
            return {"data": "not Found"}

    def delete(self, id):
        data_object = find_object_by_id(id)
        if (data_object):
            data.remove(data_object)
            return {"data": "Deleted"}
        else:
            return {"data": "not Found"}

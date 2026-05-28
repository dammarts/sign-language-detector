import os
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is not None:
        return _db
    uri = os.environ.get('MONGO_URI')
    if not uri:
        return None
    try:
        from pymongo import MongoClient
        _client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        _client.admin.command('ping')
        _db = _client['signdetect']
        return _db
    except Exception:
        return None


def create_user(username, email, password):
    db = get_db()
    if db is None:
        return None, 'Sin conexión a la base de datos'
    if db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
        return None, 'Usuario o email ya registrado'
    user_id = db.users.insert_one({
        'username': username,
        'email': email,
        'password_hash': generate_password_hash(password),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }).inserted_id
    return str(user_id), None


def authenticate_user(identifier, password):
    db = get_db()
    if db is None:
        return None, 'Sin conexión a la base de datos'
    user = db.users.find_one(
        {'$or': [{'username': identifier}, {'email': identifier}]}
    )
    if not user or not check_password_hash(user['password_hash'], password):
        return None, 'Credenciales incorrectas'
    return {'id': str(user['_id']), 'username': user['username']}, None


def save_session(user_id, session_doc):
    db = get_db()
    if db is None:
        return
    doc = dict(session_doc)
    doc['user_id'] = user_id
    session_id = doc.get('session_id')
    if session_id:
        db.sessions.update_one(
            {'session_id': session_id, 'user_id': user_id},
            {'$set': doc},
            upsert=True
        )
    else:
        doc.setdefault('created_at', datetime.now(timezone.utc).isoformat())
        db.sessions.insert_one(doc)


def get_history(user_id, country=None, limit=100):
    db = get_db()
    if db is None:
        return {'sessions': []}
    query = {'user_id': user_id}
    if country:
        query['country'] = country
    sessions = list(
        db.sessions.find(query, {'_id': 0}).sort('date', -1).limit(limit)
    )
    return {'sessions': sessions}


def delete_history(user_id, country=None):
    db = get_db()
    if db is None:
        return 0
    query = {'user_id': user_id}
    if country:
        query['country'] = country
    return db.sessions.delete_many(query).deleted_count

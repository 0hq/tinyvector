import datetime
import os
import secrets

import jwt
from dotenv import load_dotenv

load_dotenv()

def generate_jwt_secret():
    # Generate a random string of length 64 and write it to .env file
    jwt_secret = secrets.token_urlsafe(64)

    with open('.env', 'a') as file:
        file.write(f'\nJWT_SECRET={jwt_secret}')

    return jwt_secret

def generate_token(user_id):
    # Generate a token with user_id as payload with 30 days expiry
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30, minutes=0)
    }

    token = jwt.encode(payload, os.getenv('JWT_SECRET'), algorithm='HS256')

    return token

if __name__ == '__main__':
    generate_jwt_secret()
    generate_token('root')
import os
import datetime
import jwt
from dotenv import load_dotenv

load_dotenv()

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, minutes=30)
    }

    token = jwt.encode(payload, os.getenv('SECRET_KEY'), algorithm='HS256')

    return token

# Usage
print(generate_token('your_user_id')) # Replace 'your_user_id' with your actual user id

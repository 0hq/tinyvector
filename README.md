There's very simple authentication here but you probably should wrap this behind a reverse proxy (Nginx, Apache) for more security or HTTPS.
Use the generate_token script to create a new token and set a secret in .env.

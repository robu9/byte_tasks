# Private GitHub README Generator (Monorepo)

Folders:
- backend/ : Express server
- frontend/: React + Vite frontend

Steps (dev):
1. Copy backend/.env.example -> backend/.env and fill values (GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GEMINI_API_KEY, SESSION_SECRET)
2. Start backend: cd backend && npm install && npm run dev
3. Start frontend: cd frontend && npm install && npm run dev (Vite default port 5173 or adjust VITE config)

Make sure your GitHub OAuth App callback is set to the value in GITHUB_CALLBACK_URL (e.g. http://localhost:5000/auth/github/callback)

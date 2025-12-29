QUICK START (ZIP Option 1 — with Electron dev wrapper)

1) Extract this folder into your project root.
2) Create backend/.env from backend/.env.example and fill your keys.
3) Install backend deps:
   cd backend
   pip install -r requirements.txt

4) Frontend deps:
   cd ../frontend
   npm install

DEV RUN (2 windows):
- Backend:  cd backend && uvicorn main:app --reload
- Frontend: cd frontend && npm start
- Electron: cd electron && npm start  (loads http://localhost:3000, starts backend if not already)

BUILD DESKTOP (after frontend build):
- cd frontend && npm run build
- cd ../electron && npm run build
The Electron installer will package frontend/build into the app.

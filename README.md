# Voice Emotion Detection

A high-performance voice emotion detection application using Groq (Whisper + Llama) for analysis and a modern React/Next.js frontend with 3D WebGL animations.

## Deployment

This project is set up for multi-platform deployment:
- **Backend**: Deploy the root folder to [Railway](https://railway.app).
- **Frontend**: Deploy the `frontend` folder to [Vercel](https://vercel.com).

### Backend (Railway)
1. Add your repository to Railway.
2. Railway will automatically detect the `package.json` in the root and use `npm start` as the start command.
3. Configure the following Environment Variables:
   - `GROQ_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_TABLE` (default: `voice_predictions`)
   - `HF_API_KEY` (HuggingFace token for optional acoustic features)
   - `PORT` (Railway sets this automatically)

### Frontend (Vercel)
1. Import your repository into Vercel.
2. Select the `frontend` folder as the **Root Directory**.
3. Configure the following Environment Variable:
   - `NEXT_PUBLIC_API_URL`: Set this to your deployed Railway backend URL (e.g., `https://your-backend.up.railway.app`).

## Development
To run both backend and frontend locally with one command:
```bash
npm install
npm run dev
```
- Backend on: http://localhost:3000
- Frontend on: http://localhost:3001

require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const cookieSession = require('cookie-session');
const path = require('path');
const fs = require('fs');

const authRoutes = require('./routes/auth');
const repoRoutes = require('./routes/repos');
const genRoutes = require('./routes/generate');

const app = express();
app.use(helmet());
app.use(express.json({ limit: '1mb' }));
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || 'http://localhost:5173', credentials: true }));

app.use(cookieSession({
  name: 'session',
  keys: [process.env.SESSION_SECRET || 'devsecret'],
  maxAge: 24 * 60 * 60 * 1000
}));

// Ensure tmp storage exists
const tmpPath = process.env.TMP_STORAGE_PATH || path.join(__dirname, '..', '..', 'tmp');
if (!fs.existsSync(tmpPath)) fs.mkdirSync(tmpPath, { recursive: true });

app.use('/auth', authRoutes);
app.use('/api/repos', repoRoutes);
app.use('/api/generate', genRoutes);

app.get('/', (req, res) => res.json({ status: 'ok' }));

app.use((err, req, res, next) => {
  console.error('Internal server error:', err && err.message);
  res.status(500).json({ error: 'Internal server error' });
});

const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Server running on ${port}`));

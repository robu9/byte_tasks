const express = require('express');
const router = express.Router();
const axios = require('axios');

// Start OAuth flow: redirect user to GitHub's authorize URL
router.get('/github', (req, res) => {
  const clientId = process.env.GITHUB_CLIENT_ID;
  const redirectUri = process.env.GITHUB_CALLBACK_URL;
  const scope = 'repo read:org';
  const url = `https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&scope=${encodeURIComponent(scope)}`;
  return res.redirect(url);
});

// Callback: exchange code for token, store in session, redirect to frontend dashboard
router.get('/github/callback', async (req, res) => {
  const code = req.query.code;
  if (!code) return res.status(400).send('Missing code from GitHub');

  try {
    const resp = await axios.post('https://github.com/login/oauth/access_token', {
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET,
      code
    }, { headers: { Accept: 'application/json' } });

    const { access_token, error } = resp.data;
    if (error || !access_token) return res.status(400).send('OAuth failed');

    // store token in session (server-side)
    req.session.github_token = access_token;

    // Redirect to frontend dashboard
    const frontend = process.env.FRONTEND_ORIGIN || 'http://localhost:5173';
    return res.redirect(`${frontend}/dashboard`);
  } catch (err) {
    console.error('OAuth callback error:', err.message);
    return res.status(500).send('OAuth exchange failed');
  }
});

// Logout: clear session and redirect to frontend home
router.get('/logout', (req, res) => {
  req.session = null;
  const frontend = process.env.FRONTEND_ORIGIN || 'http://localhost:5173';
  res.redirect(frontend);
});

module.exports = router;

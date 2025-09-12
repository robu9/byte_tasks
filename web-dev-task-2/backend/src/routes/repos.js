const express = require('express');
const router = express.Router();
const github = require('../services/githubService');

function authMiddleware(req, res, next) {
  const token = req.session?.github_token;
  if (!token) return res.status(401).json({ error: 'Not authenticated with GitHub' });
  req.githubToken = token;
  next();
}

router.get('/', authMiddleware, async (req, res) => {
  try {
    const repos = await github.listUserRepos(req.githubToken);
    const mapped = repos.map(r => ({ id: r.id, name: r.name, full_name: r.full_name, private: r.private, owner: r.owner.login }));
    res.json(mapped);
  } catch (err) {
    if (err.response && err.response.status === 401) return res.status(401).json({ error: 'Invalid or expired GitHub token' });
    console.error('list repos error:', err.message);
    res.status(500).json({ error: 'Failed to fetch repos' });
  }
});

router.get('/:owner/:repo/contents', authMiddleware, async (req, res) => {
  const { owner, repo } = req.params;
  try {
    const contents = await github.getRepoContents(req.githubToken, owner, repo);
    res.json(contents);
  } catch (err) {
    if (err.response && err.response.status === 401) return res.status(401).json({ error: 'Invalid or expired GitHub token' });
    console.error('get contents error:', err.message);
    res.status(500).json({ error: 'Failed to fetch repo contents' });
  }
});

module.exports = router;

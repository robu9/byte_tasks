// backend/src/routes/readmeRoute.js
const express = require('express');
const router = express.Router();
const { getRepo, getFileContent, createOrUpdateFile } = require('../services/githubService');
const { generateReadme } = require('../services/geminiClient');

router.post('/generate', async (req, res) => {
  try {
    const { owner, repo } = req.body;
    const token = req.session.github_token; // OAuth token stored in session

    if (!token) return res.status(401).json({ error: 'Not authenticated with GitHub' });

    // Fetch repo data and existing README
    const repoData = await getRepo(token, owner, repo);
    const readmeContent = await getFileContent(token, owner, repo, 'README.md');

    // Generate README with Gemini
    const generatedReadme = await generateReadme(repoData, readmeContent);

    // Save generated README back to GitHub
    await createOrUpdateFile(token, owner, repo, 'README.md', generatedReadme);

    res.json({ message: 'README generated and saved successfully', readme: generatedReadme });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;

// backend/src/routes/generate.js
const express = require('express');
const router = express.Router();
const github = require('../services/githubService');
const { generateReadme } = require('../services/geminiClient');

function authMiddleware(req, res, next) {
  const token = req.session?.github_token;
  if (!token) return res.status(401).json({ error: 'Not authenticated with GitHub' });
  req.githubToken = token;
  next();
}

// Build prompt using actual repo data and key files
async function buildPrompt(token, owner, repo) {
  const repoMeta = await github.getRepo(token, owner, repo);

  const keyFiles = ['package.json','requirements.txt','pyproject.toml','Pipfile','Dockerfile','README.md'];
  const filesData = {};
  for (const f of keyFiles) {
    try {
      const content = await github.getFileContent(token, owner, repo, f);
      if (content) filesData[f] = content.slice(0, 2000); // limit to first 2k chars
    } catch (e) {
      // ignore missing files
    }
  }

  // Top-level file/folder summary
  let treeSummary = '';
  try {
    const contents = await github.getRepoContents(token, owner, repo, '');
    if (Array.isArray(contents)) {
      treeSummary = contents.slice(0,50).map(c => `${c.type}:${c.name}`).join(', ');
    }
  } catch (e) {}

  // Build prompt
  let prompt = `
You are an expert technical writer. Generate a professional GitHub README.md in Markdown for the repository "${repo}" owned by "${owner}".
Use the actual repository metadata and file contents provided.

Repository description: ${repoMeta.description || 'None'}
Top-level files: ${treeSummary}

`;
  for (const [fname, fcontent] of Object.entries(filesData)) {
    prompt += `File: ${fname}\n${fcontent}\n\n`;
  }

  prompt += `
Required sections: Title, Description, Key Features, Installation Instructions, Technology Stack, Project Structure, Usage Examples, License.
Use actual repo information. Do not invent features or data.
Output only Markdown.
`;

  return { prompt, repoMeta };
}

router.post('/readme', authMiddleware, async (req, res) => {
  const { owner, repo, writeToRepo } = req.body || {};
  if (!owner || !repo) return res.status(400).json({ error: 'owner and repo required' });

  try {
    // Build prompt
    const { prompt, repoMeta } = await buildPrompt(req.githubToken, owner, repo);

    // Generate README with Gemini
    const aiText = await generateReadme(repoMeta, prompt);
    if (!aiText) return res.status(500).json({ error: 'Gemini returned empty response' });

    // Optionally write back to repo
    if (writeToRepo) {
      try {
        await github.createOrUpdateFile(req.githubToken, owner, repo, 'README.md', aiText, 'docs: add AI-generated README');
        return res.json({ generated: aiText, wroteToRepo: true });
      } catch (err) {
        console.error('write README error:', err.message);
        return res.status(500).json({ error: 'Failed to write README to repository' });
      }
    }

    res.json({ generated: aiText, wroteToRepo: false });
  } catch (err) {
    if (err.response && err.response.status === 401) return res.status(401).json({ error: 'Invalid or expired GitHub token' });
    console.error('generate readme error:', err.message);
    res.status(500).json({ error: 'Failed to generate README' });
  }
});

module.exports = router;

const axios = require('axios');

// Helper to build GitHub headers safely
function buildHeaders(token) {
  if (!token) throw new Error('GitHub token is missing or invalid');
  return {
    Authorization: `token ${token}`.trim(),
    Accept: 'application/vnd.github+json'
  };
}

async function listUserRepos(token) {
  const resp = await axios.get('https://api.github.com/user/repos?per_page=200', {
    headers: buildHeaders(token)
  });
  return resp.data;
}

async function getRepo(token, owner, repo) {
  const resp = await axios.get(`https://api.github.com/repos/${owner}/${repo}`, {
    headers: buildHeaders(token)
  });
  return resp.data;
}

async function getRepoContents(token, owner, repo, path = '') {
  const url = path
    ? `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(path)}`
    : `https://api.github.com/repos/${owner}/${repo}/contents`;
  const resp = await axios.get(url, { headers: buildHeaders(token) });
  return resp.data;
}

async function getFileContent(token, owner, repo, path) {
  try {
    const resp = await axios.get(
      `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(path)}`,
      { headers: buildHeaders(token) }
    );
    if (resp.data && resp.data.content) {
      return Buffer.from(resp.data.content, resp.data.encoding || 'base64').toString('utf8');
    }
    return null;
  } catch (err) {
    if (err.response && err.response.status === 404) return null;
    throw err;
  }
}

async function createOrUpdateFile(token, owner, repo, path, content, message = 'docs: add AI-generated README') {
  let sha = null;
  try {
    const getResp = await axios.get(
      `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(path)}`,
      { headers: buildHeaders(token) }
    );
    sha = getResp.data.sha;
  } catch (err) {
    if (!(err.response && err.response.status === 404)) throw err;
  }

  const payload = {
    message,
    content: Buffer.from(content, 'utf8').toString('base64')
  };
  if (sha) payload.sha = sha;

  const resp = await axios.put(
    `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(path)}`,
    payload,
    { headers: buildHeaders(token) }
  );
  return resp.data;
}

module.exports = { listUserRepos, getRepo, getRepoContents, getFileContent, createOrUpdateFile };

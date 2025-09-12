const fs = require('fs');
const path = require('path');
const sanitize = require('sanitize-filename');

function safeWriteFile(baseDir, filename, data) {
  const safeName = sanitize(filename);
  const fullPath = path.resolve(baseDir, safeName);
  if (!fullPath.startsWith(path.resolve(baseDir))) throw new Error('Path traversal attempt');
  fs.writeFileSync(fullPath, data, { encoding: 'utf8' });
  return fullPath;
}

module.exports = { safeWriteFile };

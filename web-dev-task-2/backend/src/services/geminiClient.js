const { GoogleGenerativeAI } = require('@google/generative-ai');

if (!process.env.GEMINI_API_KEY) {
  console.error('GEMINI_API_KEY not set in .env');
  process.exit(1);
}

// Correctly initialize the client by passing the API key directly
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

async function generateReadme(repoData, readmeContent = '') {
  try {
    // Select the generative model and apply generation configuration
    const model = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash', // Using a stable and recommended model
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 1000,
      }
    });

    const prompt = `
You are a highly skilled AI assistant specialized in generating professional GitHub README files.
Follow these sections exactly:
1. Project Title
2. Description
3. Key Features
4. Installation Instructions
5. Technology Stack
6. Project Structure
7. Usage Examples
8. License Information

Repository Information:
- Name: ${repoData.name}
- Description: ${repoData.description || 'No description provided'}
- Languages: ${(repoData.languages || [repoData.language] || []).filter(Boolean).join(', ') || 'Unknown'}
- License: ${repoData.license?.name || 'Unspecified'}
- Existing README Content: ${readmeContent || 'None'}

Output only the README in Markdown format, concise and professional.
`;

    // Generate content using the new method
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    return text || 'Failed to generate README: empty response';
  } catch (err) {
    console.error('generate readme error:', err.message);
    throw new Error('Failed to generate README: ' + err.message);
  }
}

module.exports = { generateReadme };
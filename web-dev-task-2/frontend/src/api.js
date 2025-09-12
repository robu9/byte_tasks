import axios from 'axios';
const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || 'http://localhost:5000';
export const api = axios.create({ baseURL: API_ORIGIN, withCredentials: true });
export function startOAuth() { window.location.href = `${API_ORIGIN}/auth/github`; }
export function listRepos() { return api.get('/api/repos'); }
export function generateReadme(payload) { return api.post('/api/generate/readme', payload); }
export function logout() { window.location.href = '/auth/logout'; }
export function saveReadme(payload) { return api.post('/api/generate/readme', payload); }

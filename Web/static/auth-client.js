/**
 * LawPakAI — Client-side Auth Module
 * Manages JWT tokens, auto-refresh, and API calls.
 * Include this in every authenticated page.
 */

const AUTH_API = '/api';

const AuthClient = {
  // ── Token Storage (in-memory, not localStorage — report Section 9) ──
  _accessToken: null,
  _refreshToken: null,
  _user: null,

  init() {
    // On page load, try to restore from sessionStorage (survives refresh, not new tab)
    const saved = sessionStorage.getItem('lawpakai_auth');
    if (saved) {
      try {
        const data = JSON.parse(saved);
        this._accessToken = data.access_token;
        this._refreshToken = data.refresh_token;
        this._user = data.user;
      } catch (e) { /* ignore */ }
    }
  },

  isLoggedIn() {
    return !!this._accessToken;
  },

  getUser() {
    return this._user;
  },

  _save(data) {
    this._accessToken = data.access_token;
    this._refreshToken = data.refresh_token;
    this._user = data.user;
    sessionStorage.setItem('lawpakai_auth', JSON.stringify({
      access_token: data.access_token,
      refresh_token: data.refresh_token,
      user: data.user,
    }));
  },

  _clear() {
    this._accessToken = null;
    this._refreshToken = null;
    this._user = null;
    sessionStorage.removeItem('lawpakai_auth');
  },

  // ── Register ──
  async register(fullName, email, password) {
    const res = await fetch(`${AUTH_API}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_name: fullName, email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Registration failed');
    this._save(data);
    return data;
  },

  // ── Login ──
  async login(email, password) {
    const res = await fetch(`${AUTH_API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Login failed');
    this._save(data);
    return data;
  },

  // ── Logout ──
  async logout() {
    try {
      await this.apiFetch('/auth/logout', {
        method: 'POST',
        body: JSON.stringify({ refresh_token: this._refreshToken }),
      });
    } catch (e) { /* ignore — clear locally anyway */ }
    this._clear();
    window.location.href = '/register.html';
  },

  // ── Auto-refresh access token ──
  async refreshAccessToken() {
    if (!this._refreshToken) return false;
    try {
      const res = await fetch(`${AUTH_API}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: this._refreshToken }),
      });
      if (!res.ok) {
        this._clear();
        return false;
      }
      const data = await res.json();
      this._accessToken = data.access_token;
      // Update sessionStorage
      const saved = JSON.parse(sessionStorage.getItem('lawpakai_auth') || '{}');
      saved.access_token = data.access_token;
      sessionStorage.setItem('lawpakai_auth', JSON.stringify(saved));
      return true;
    } catch (e) {
      this._clear();
      return false;
    }
  },

  // ── Authenticated API fetch (auto-retries on 401 with refresh) ──
  async apiFetch(path, options = {}) {
    const url = `${AUTH_API}${path}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };

    if (this._accessToken) {
      headers['Authorization'] = `Bearer ${this._accessToken}`;
    }

    let res = await fetch(url, { ...options, headers });

    // If 401, try refresh and retry once
    if (res.status === 401 && this._refreshToken) {
      const refreshed = await this.refreshAccessToken();
      if (refreshed) {
        headers['Authorization'] = `Bearer ${this._accessToken}`;
        res = await fetch(url, { ...options, headers });
      }
    }

    return res;
  },

  // ── Require auth (redirect to register if not logged in) ──
  requireAuth() {
    if (!this.isLoggedIn()) {
      window.location.href = '/register.html';
      return false;
    }
    return true;
  },
};

// Auto-init on load
AuthClient.init();

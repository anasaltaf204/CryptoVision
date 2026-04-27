/**
 * services/api.js
 *
 * Axios instance for the internal prediction backend.
 *
 * Changes from original:
 *  - Adds a request interceptor that injects an X-Request-ID header for
 *    easier backend log correlation.
 *  - Response interceptor now handles network-level errors (no response
 *    object) separately from HTTP error responses, and exposes a typed
 *    ApiError so callers can branch on err.type.
 *  - Timeout is overridden per-request (120s only for /api/predict).
 */

import axios from "axios";

// ── Custom error class ────────────────────────────────────────────────────────
export class ApiError extends Error {
    constructor(message, type = "unknown", status = null) {
        super(message);
        this.name = "ApiError";
        this.type = type;   // "network" | "server" | "timeout" | "unknown"
        this.status = status; // HTTP status code or null
    }
}

// ── Axios instance ────────────────────────────────────────────────────────────
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
    headers: { "Content-Type": "application/json" },
    timeout: 15_000, // 15 s default; overridden per-request below
});

// ── Request interceptor ───────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
    // Model inference is slow on cold start — give it 2 minutes
    if (config.url?.includes("/api/predict")) {
        config.timeout = 120_000;
    }
    // Unique ID for backend log correlation
    config.headers["X-Request-ID"] = crypto.randomUUID();
    return config;
});

// ── Response interceptor ─────────────────────────────────────────────────────
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.code === "ECONNABORTED" || error.code === "ERR_CANCELED") {
            return Promise.reject(
                new ApiError("Request timed out. Please try again.", "timeout")
            );
        }
        if (!error.response) {
            return Promise.reject(
                new ApiError("Cannot reach the server. Check your connection.", "network")
            );
        }
        const { status, data } = error.response;
        const message = data?.detail || data?.message || `Server error (${status})`;
        return Promise.reject(new ApiError(message, "server", status));
    }
);

export default api;

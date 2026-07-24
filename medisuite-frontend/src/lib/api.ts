import axios from "axios";

import { useAuthStore } from "@/store/authStore";

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
});

// Attach the JWT (if any) as a Bearer token on every request.
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// On any 401, clear the session and send the user to /login (except when already
// on a public page, so we don't bounce the landing/auth pages).
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      const { token, logout } = useAuthStore.getState();
      if (token) logout();
      if (typeof window !== "undefined") {
        const path = window.location.pathname;
        if (path !== "/" && path !== "/login" && path !== "/register") {
          window.location.href = "/login";
        }
      }
    }
    return Promise.reject(error);
  }
);

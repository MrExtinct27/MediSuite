import { create } from "zustand";

export interface AuthUser {
  id?: number;
  username: string;
}

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  /** True once the store has read localStorage on the client. Used to avoid a
   *  flash of protected content / login buttons before the session is known. */
  hydrated: boolean;
  login: (token: string, user: AuthUser) => void;
  logout: () => void;
  hydrate: () => void;
}

const TOKEN_KEY = "medisuite_token";
const USER_KEY = "medisuite_user";

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  hydrated: false,

  login: (token, user) => {
    if (typeof window !== "undefined") {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
    set({ token, user, isAuthenticated: true });
  },

  logout: () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
    set({ token: null, user: null, isAuthenticated: false });
  },

  // Called once on the client after mount to restore the session from localStorage.
  hydrate: () => {
    if (typeof window === "undefined") {
      set({ hydrated: true });
      return;
    }
    try {
      const token = localStorage.getItem(TOKEN_KEY);
      const rawUser = localStorage.getItem(USER_KEY);
      const user = rawUser ? (JSON.parse(rawUser) as AuthUser) : null;
      set({ token: token ?? null, user, isAuthenticated: !!token, hydrated: true });
    } catch {
      set({ hydrated: true });
    }
  },
}));

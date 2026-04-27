/**
 * store/slices/userSlice.js
 *
 * Redux slice for the authenticated user's extended profile data.
 *
 * Why Redux here (and not just AuthContext)?
 *  - AuthContext gives us the raw Firebase Auth object (uid, displayName,
 *    photoURL). That's authentication identity, not application profile data.
 *  - The Firestore "users" document holds richer fields: bio, updatedAt, etc.
 *  - Multiple components read the profile: Navbar (avatar + name),
 *    ProfilePage (full edit form), future pages like watchlists.
 *  - Without Redux, each would trigger a separate Firestore read.
 *
 * Boundary:
 *  - AuthContext  → Firebase Auth state (login / logout / onAuthStateChanged)
 *  - userSlice    → Firestore profile data + optimistic UI updates
 */

import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import {
    getUserProfile,
    updateUserProfile,
} from "../../services/userService";

// ── Thunks ────────────────────────────────────────────────────────────────────

/** Load the full Firestore profile for the authenticated user. */
export const loadUserProfile = createAsyncThunk(
    "user/loadProfile",
    async (uid, { rejectWithValue }) => {
        try {
            return await getUserProfile(uid);
        } catch (err) {
            return rejectWithValue(err.message);
        }
    }
);

/** Save profile changes and update Firebase Auth in one shot. */
export const saveUserProfile = createAsyncThunk(
    "user/saveProfile",
    async ({ uid, updates }, { rejectWithValue }) => {
        try {
            return await updateUserProfile(uid, updates);
        } catch (err) {
            return rejectWithValue(err.message);
        }
    }
);

// ── Slice ─────────────────────────────────────────────────────────────────────

const userSlice = createSlice({
    name: "user",
    initialState: {
        profile:     null,   // Firestore document data
        status:      "idle", // "idle" | "loading" | "ready" | "error"
        saveStatus:  "idle", // "idle" | "saving" | "saved" | "error"
        error:       null,
        saveError:   null,
    },
    reducers: {
        // Called on logout — clears the cached profile immediately
        clearProfile(state) {
            state.profile    = null;
            state.status     = "idle";
            state.saveStatus = "idle";
            state.error      = null;
            state.saveError  = null;
        },
        // Reset save banner after showing the success message
        resetSaveStatus(state) {
            state.saveStatus = "idle";
            state.saveError  = null;
        },
    },
    extraReducers: (builder) => {
        // ── Load profile ───────────────────────────────────────────────────
        builder
            .addCase(loadUserProfile.pending, (state) => {
                state.status = "loading";
                state.error  = null;
            })
            .addCase(loadUserProfile.fulfilled, (state, { payload }) => {
                state.profile = payload;
                state.status  = "ready";
            })
            .addCase(loadUserProfile.rejected, (state, { payload }) => {
                state.status = "error";
                state.error  = payload;
            });

        // ── Save profile ───────────────────────────────────────────────────
        builder
            .addCase(saveUserProfile.pending, (state) => {
                state.saveStatus = "saving";
                state.saveError  = null;
            })
            .addCase(saveUserProfile.fulfilled, (state, { payload }) => {
                // Merge the returned updates into the cached profile
                state.profile    = { ...state.profile, ...payload };
                state.saveStatus = "saved";
            })
            .addCase(saveUserProfile.rejected, (state, { payload }) => {
                state.saveStatus = "error";
                state.saveError  = payload;
            });
    },
});

export const { clearProfile, resetSaveStatus } = userSlice.actions;
export default userSlice.reducer;

// ── Selectors ─────────────────────────────────────────────────────────────────
export const selectUserProfile   = (s) => s.user.profile;
export const selectUserStatus    = (s) => s.user.status;
export const selectUserSaveState = (s) => ({
    saveStatus: s.user.saveStatus,
    saveError:  s.user.saveError,
});

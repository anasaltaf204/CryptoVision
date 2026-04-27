/**
 * context/AuthContext.jsx
 *
 * Changes in this version:
 *  - On login / auth state change, dispatches loadUserProfile to Redux so
 *    the Firestore profile is loaded once and shared everywhere.
 *  - On logout, dispatches clearProfile to wipe the cached profile.
 *  - useAuth() now also exposes `updateUserData` for rare cases where
 *    a component needs to force-refresh the Firebase Auth object.
 */

import { createContext, useEffect, useContext, useState } from "react";
import { useDispatch } from "react-redux";
import { auth, db } from "../services/firebase";
import {
    onAuthStateChanged,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    updateProfile,
} from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { loadUserProfile, clearProfile } from "../store/slices/userSlice";

const AuthContext = createContext(null);

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used inside <AuthContextProvider>");
    return ctx;
}

export function AuthContextProvider({ children }) {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading]         = useState(true);
    const dispatch                      = useDispatch();

    async function signup(email, password, username) {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        await updateProfile(credential.user, { displayName: username });
        await setDoc(doc(db, "users", credential.user.uid), {
            username,
            email,
            uid:      credential.user.uid,
            photoURL: null,
            bio:      "",
        });
        return credential;
    }

    function login(email, password) {
        return signInWithEmailAndPassword(auth, email, password);
    }

    async function logout() {
        await signOut(auth);
        dispatch(clearProfile());
    }

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            setCurrentUser(user);
            setLoading(false);

            if (user) {
                // Load the extended Firestore profile into Redux on every
                // auth state change (login, page refresh with active session)
                dispatch(loadUserProfile(user.uid));
            }
        });
        return unsubscribe;
    }, [dispatch]);

    const value = { currentUser, signup, login, logout, loading };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
}

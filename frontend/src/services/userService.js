/**
 * services/userService.js
 *
 * All user-profile Firestore operations live here.
 * Components and hooks never import firebase/firestore directly.
 *
 * Storage strategy:
 *  - Profile text data  → Firestore "users/{uid}" document
 *  - Profile photo      → Firebase Storage "avatars/{uid}" (base64 data-URL
 *    stored directly in Firestore for simplicity; swap to Storage SDK if
 *    avatars exceed a few hundred KB in production)
 */

import {
    doc,
    getDoc,
    setDoc,
    updateDoc,
    serverTimestamp,
} from "firebase/firestore";
import { updateProfile } from "firebase/auth";
import { auth, db } from "./firebase";

/**
 * Fetches the Firestore profile document for the given uid.
 * Falls back gracefully if the document doesn't exist yet (legacy accounts).
 * @returns {Promise<object|null>}
 */
export async function getUserProfile(uid) {
    const ref  = doc(db, "users", uid);
    const snap = await getDoc(ref);
    return snap.exists() ? snap.data() : null;
}

/**
 * Creates or completely replaces the Firestore profile document.
 * Called once at sign-up.
 */
export async function createUserProfile(uid, data) {
    const ref = doc(db, "users", uid);
    await setDoc(ref, {
        ...data,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
    });
}

/**
 * Partially updates the Firestore profile document AND syncs the
 * Firebase Auth displayName/photoURL fields so both stay in sync.
 *
 * @param {string} uid
 * @param {{ username?: string, bio?: string, photoURL?: string }} updates
 */
export async function updateUserProfile(uid, updates) {
    // 1. Update Firestore document
    const ref = doc(db, "users", uid);
    await updateDoc(ref, {
        ...updates,
        updatedAt: serverTimestamp(),
    });

    // 2. Mirror name / avatar onto the Firebase Auth profile so
    //    currentUser.displayName and currentUser.photoURL stay fresh
    const authUpdates = {};
    if (updates.username  !== undefined) authUpdates.displayName = updates.username;
    if (updates.photoURL  !== undefined) authUpdates.photoURL    = updates.photoURL;

    if (Object.keys(authUpdates).length > 0 && auth.currentUser) {
        await updateProfile(auth.currentUser, authUpdates);
    }

    return { uid, ...updates };
}
